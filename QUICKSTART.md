# 🚀 Quick Start Guide

Get your handwriting digit detector running in 5 minutes!

## Option 1: Windows (Easiest)

### 1. Install Python Dependencies
Open PowerShell in the project folder and run:
```powershell
pip install -r requirements.txt
```

### 2. Verify Setup
```powershell
python setup_check.py
```
You should see all checkmarks ✅

### 3. Train the Model
```powershell
python train_model.py
```
Wait 5-15 minutes for training to complete. You'll see progress like:
```
Epoch [1/10] - Train Loss: 0.2345, Test Loss: 0.1234, Accuracy: 96.50%
Epoch [2/10] - Train Loss: 0.1123, Test Loss: 0.0987, Accuracy: 97.20%
...
Training complete! Model saved and ready for use.
```

### 4. Launch the App
```powershell
streamlit run app.py
```

Your browser will open automatically to `http://localhost:8501`

---

## Option 2: Using Virtual Environment (Recommended)

### 1. Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Setup
```powershell
python setup_check.py
```

### 4. Train the Model
```powershell
python train_model.py
```

### 5. Run the App
```powershell
streamlit run app.py
```

### To Deactivate Virtual Environment Later
```powershell
deactivate
```

---

## What Each Script Does

### 📊 `setup_check.py`
Verifies that:
- All Python packages are installed
- MNIST data files are present
- Everything is ready to go

### 🎓 `train_model.py`
Trains the CNN model:
- Loads MNIST dataset (60,000 training images)
- Trains for 10 epochs
- Saves model to `model/digit_detector.pth`
- Shows accuracy metrics

### 🌐 `app.py`
Web interface with:
- **Upload Tab**: Upload image files and get predictions
- **Draw Tab**: Draw digits and predict in real-time
- **About Tab**: Learn about the model

---

## Using the Application

### Upload Image Tab
1. Click "Choose an image file"
2. Select a JPG, PNG, or BMP file with a handwritten digit
3. See the prediction and confidence

### Draw Digit Tab
1. Draw a digit (0-9) on the white canvas
2. Click "🔍 Predict Digit"
3. View results

### Keyboard Shortcuts
- **Clear Canvas**: Refresh the page or click outside
- **Undo**: There's no undo, just redraw

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```powershell
pip install -r requirements.txt
```

### "Model not found" when running app
Make sure you completed step 3 (Training the model)

### App runs slowly on first prediction
Normal! PyTorch needs to initialize. Subsequent predictions are instant.

### Low accuracy on my handwriting
The model is trained on MNIST (printed digits). Write clearly and centered.

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Dependencies won't install | Try: `pip install --upgrade pip` first |
| CUDA errors | The model defaults to CPU - it's fine, just slower |
| Port 8501 already in use | Run: `streamlit run app.py --server.port 8502` |
| Out of memory during training | Reduce batch size in `train_model.py` (change `batch_size = 64` to `32`) |

---

## Performance Tips

### For Faster Training
- Use GPU (if available): Model will automatically use CUDA
- Reduce epochs in `train_model.py`

### For Better Accuracy
- Train for more epochs (increase in `train_model.py`)
- Use consistent handwriting style
- Write digits that fill the space

### For Faster Predictions
- Predictions are already instant! ⚡
- First prediction loads model (1-2 seconds)
- Subsequent predictions: <100ms

---

## Next Steps

After getting the app running:

1. ✅ Test with sample images
2. ✅ Try drawing digits in the Draw tab
3. ✅ Check confidence scores
4. ✅ Experiment with different writing styles
5. ✅ Read the README.md for more details

---

## File Structure After Training

```
Hand-writting Detector/
├── app.py
├── train_model.py
├── utils.py
├── setup_check.py
├── requirements.txt
├── README.md
├── QUICKSTART.md (this file)
├── .gitignore
├── model/
│   └── digit_detector.pth  ← Generated after training
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

---

## Need Help?

1. Check `README.md` for detailed information
2. Check `setup_check.py` output for missing components
3. Read error messages - they usually tell you what's wrong
4. Code is well-commented - read the source files!

---

**Happy digit detecting! 🎉**

Questions? Check the comments in the Python files - they explain everything!
