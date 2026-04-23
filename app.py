import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from train_model import SimpleDigitDetector
import os


@st.cache_resource
def load_model():
    """
    Load the trained model
    """
    model = SimpleDigitDetector()
    model_path = 'model/digit_detector.pth'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Please train the model first by running: python train_model.py")
        st.stop()
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image):
    """
    Preprocess image for the model with better contrast handling
    """
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy
    image_array = np.array(image).astype(np.float32)
    
    # Auto-invert if needed (handle both light-on-dark and dark-on-light)
    if image_array.mean() > 127:
        image_array = 255 - image_array
    
    # Normalize
    image_array = image_array / 255.0
    
    # Add batch and channel dimensions (1, 1, 28, 28)
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_array, image_tensor


def predict(model, image_tensor):
    """
    Make prediction on the image
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_digit = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_digit].item() * 100
    
    return predicted_digit, confidence, probabilities[0].numpy()


def segment_digits(image_data):
    """
    Segment multiple digits from canvas/image using contour detection
    Returns list of (digit_image, bounding_box)
    """
    # Convert to numpy if PIL Image
    if isinstance(image_data, Image.Image):
        img_array = np.array(image_data)
    else:
        img_array = image_data.astype(np.uint8)
    
    # Convert RGBA to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if background is light
    if binary.mean() > 127:
        binary = 255 - binary
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    digit_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Only keep significant contours (avoid noise)
        if area > 100 and w > 10 and h > 10:
            digit_regions.append((x, y, w, h))
    
    # Sort by x position (left to right)
    digit_regions.sort(key=lambda r: r[0])
    
    return digit_regions, gray, binary


def crop_and_preprocess_digit(image_array, x, y, w, h, padding=10):
    """
    Crop digit region and preprocess for model
    """
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_array.shape[1] - x, w + 2 * padding)
    h = min(image_array.shape[0] - y, h + 2 * padding)
    
    # Crop
    digit_img = image_array[y:y+h, x:x+w]
    
    # Convert to PIL and preprocess
    digit_pil = Image.fromarray((digit_img).astype(np.uint8))
    
    # Resize to 28x28
    digit_pil = ImageOps.grayscale(digit_pil)
    digit_pil = digit_pil.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize
    digit_array = np.array(digit_pil).astype(np.float32)
    
    # Auto-invert if needed
    if digit_array.mean() > 127:
        digit_array = 255 - digit_array
    
    digit_array = digit_array / 255.0
    digit_tensor = torch.from_numpy(digit_array).unsqueeze(0).unsqueeze(0)
    
    return digit_array, digit_tensor


# Page configuration
st.set_page_config(
    page_title="Handwriting Digit Detector",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Handwriting Digit Detector")
st.markdown("---")

# Load model
model = load_model()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "✏️ Draw Digits", "📊 About"])

# Tab 1: Upload Image
with tab1:
    st.subheader("Upload a handwritten digit image")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    with col2:
        detect_multiple = st.checkbox("Detect multiple digits", value=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image")
        
        with col2:
            if detect_multiple:
                # Segment and detect multiple digits
                img_array = np.array(image)
                digit_regions, gray, binary = segment_digits(image)
                
                if len(digit_regions) == 0:
                    st.warning("⚠️ No digits detected. Try uploading an image with clearer digits.")
                else:
                    st.markdown(f"### 🔍 Found {len(digit_regions)} digit(s)")
                    
                    predictions = []
                    cols = st.columns(min(4, len(digit_regions)))
                    
                    for idx, (x, y, w, h) in enumerate(digit_regions):
                        digit_array, digit_tensor = crop_and_preprocess_digit(gray, x, y, w, h)
                        digit, confidence, probs = predict(model, digit_tensor)
                        predictions.append((digit, confidence))
                        
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            st.markdown(f"**Position {idx + 1}**")
                            st.markdown(f"<h2 style='text-align: center; color: #2ecc71;'>{digit}</h2>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center; color: #3498db;'>{confidence:.1f}%</p>", unsafe_allow_html=True)
                            st.image(digit_array, width=80)
                    
                    # Show full number
                    full_number = ''.join([str(d[0]) for d in predictions])
                    st.markdown("---")
                    st.markdown(f"### 📊 Full Number: {full_number}")
            else:
                # Single digit detection
                image_array, image_tensor = preprocess_image(image)
                digit, confidence, probabilities = predict(model, image_tensor)
                
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f"<h1 style='text-align: center; color: #2ecc71;'>{digit}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #3498db;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
                
                st.markdown("### 📈 Confidence by Digit")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart({f"Digit {i}": probabilities[i] * 100 for i in range(10)})
                with col2:
                    st.text("Probabilities:")
                    for i in range(10):
                        st.write(f"  {i}: {probabilities[i]*100:.1f}%")

# Tab 2: Draw Digits
with tab2:
    st.subheader("Draw one or more digits on the canvas")
    st.info("💡 Draw each digit clearly. The system will automatically detect and segment them.")
    
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data, caption="Your Drawing")
        
        with col2:
            if canvas_result.image_data is not None and st.button("🔍 Recognize Digits", type="primary"):
                image_data = canvas_result.image_data.astype(np.uint8)
                digit_regions, gray, binary = segment_digits(image_data)
                
                if len(digit_regions) == 0:
                    st.warning("⚠️ No digits detected. Please draw clearer digits.")
                else:
                    st.markdown(f"### 🔍 Detected {len(digit_regions)} digit(s)")
                    
                    predictions = []
                    cols = st.columns(min(5, len(digit_regions)))
                    
                    for idx, (x, y, w, h) in enumerate(digit_regions):
                        digit_array, digit_tensor = crop_and_preprocess_digit(gray, x, y, w, h)
                        digit, confidence, probs = predict(model, digit_tensor)
                        predictions.append((digit, confidence))
                        
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            st.markdown(f"**Digit {idx + 1}**")
                            st.markdown(f"<h2 style='text-align: center; color: #2ecc71;'>{digit}</h2>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center; font-size: 14px;'>{confidence:.1f}%</p>", unsafe_allow_html=True)
                            st.image(digit_array, width=70)
                    
                    # Show full number
                    full_number = ''.join([str(d[0]) for d in predictions])
                    st.markdown("---")
                    st.markdown(f"### 📊 Recognized Number: {full_number}")
                    
                    # Show segmentation
                    with st.expander("📸 Show segmentation details"):
                        vis_image = gray.copy()
                        for i, (x, y, w, h) in enumerate(digit_regions):
                            cv2.rectangle(vis_image, (x, y), (x+w, y+h), 128, 2)
                            cv2.putText(vis_image, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
                        
                        st.image(vis_image, caption="Detected digit regions", use_column_width=True)
            
            st.markdown("---")
            st.markdown("**How to use:**")
            st.markdown("1. Draw digits (0-9) on the black canvas")
            st.markdown("2. Click 'Recognize Digits' to see results")
            st.markdown("3. The model will automatically segment and predict each digit")
    
    except ImportError:
        st.error("⚠️ Drawing tool not installed. Install with: pip install streamlit-drawable-canvas")

# Tab 3: About
with tab3:
    st.subheader("About This Application")
    
    st.markdown("""
    ### 🎯 What is this?
    This is a **Handwriting Digit Detector** - an AI application that recognizes handwritten digits (0-9) 
    using a trained deep learning model. It now supports **multiple digit detection**!
    
    ### 🧠 How it works
    - **Dataset**: Trained on the MNIST dataset (180,000 augmented handwritten digit images)
    - **Model**: Improved Convolutional Neural Network with 3 convolutional layers and 3 fully connected layers
    - **Accuracy**: ~99% on test data
    - **Multi-digit**: Uses contour detection to segment and recognize multiple digits
    
    ### 🏗️ Improved Model Architecture
    ```
    Input (28x28 grayscale image)
         ↓
    Conv2d (32 filters) + BatchNorm → ReLU → MaxPool2d
         ↓
    Conv2d (64 filters) + BatchNorm → ReLU → MaxPool2d
         ↓
    Conv2d (128 filters) + BatchNorm → ReLU → MaxPool2d
         ↓
    Flatten
         ↓
    Linear (256 units) → ReLU → Dropout(0.5)
         ↓
    Linear (128 units) → ReLU → Dropout(0.3)
         ↓
    Linear (10 units) - Output predictions
    ```
    
    ### ✨ Features
    - **Upload Images**: Test with your own handwritten digit images
    - **Draw & Predict**: Draw digits directly on the canvas for instant prediction
    - **Multiple Digits**: Automatically detects and segments multiple digits from a single image
    - **Confidence Scores**: See how confident the model is about each digit
    - **Segmentation Visualization**: View how digits were detected and separated
    
    ### 🔧 Technologies Used
    - **PyTorch**: Deep learning framework
    - **OpenCV**: Image segmentation and contour detection
    - **Streamlit**: Web interface
    - **Pillow**: Image processing
    
    ### 📊 What's Improved
    - **Better Model**: 3 convolutional layers with batch normalization
    - **Data Augmentation**: 180,000 training samples (rotation + translation)
    - **Multi-digit Support**: Automatic digit segmentation using contour detection
    - **Better Preprocessing**: Auto-invert colors to handle different backgrounds
    - **Learning Rate Scheduling**: Reduces learning rate during training for better convergence
    
    """)
    
    st.markdown("---")
    st.info("💡 **Tips for best results:**\n"
            "- Write digits clearly and separated\n"
            "- Use good contrast (dark on light or light on dark)\n"
            "- Write digits of similar size\n"
            "- The model works best with digits in the center\n"
            "- Multiple digits: space them apart for better segmentation")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 12px;'>
    Made with ❤️ using Streamlit | Handwriting Digit Detector v2.0 | Multi-digit Support ✨
</div>
""", unsafe_allow_html=True)
