import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from utils import load_mnist_data
import os


class SimpleDigitDetector(nn.Module):
    """
    Improved Convolutional Neural Network for digit detection
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> FC
    """
    def __init__(self):
        super(SimpleDigitDetector, self).__init__()
        
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces 28x28 to 14x14
        
        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces 14x14 to 7x7
        
        # Third convolutional layer for better feature extraction
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # Reduces 7x7 to 3x3
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)  # 10 digits (0-9)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def train_model(train_loader, test_loader, epochs=15, learning_rate=0.001):
    """
    Train the digit detector model with improved architecture
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleDigitDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, 'model/digit_detector.pth')
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}% ⭐ (Best)")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
        
        scheduler.step()
    
    return model


def save_model(model, filepath='model/digit_detector.pth'):
    """
    Save trained model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def main():
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist_data('.')
    
    # Reshape images to add channel dimension (N, 1, 28, 28)
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)
    
    # Convert to PyTorch tensors
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()
    
    # Data augmentation transforms
    augmentation = transforms.Compose([
        transforms.RandomRotation(10),      # Rotate by up to 10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
    ])
    
    # Apply augmentation to training data
    augmented_images_1 = []
    augmented_images_2 = []
    for img in train_images:
        img_pil = transforms.ToPILImage()(img)
        augmented_images_1.append(transforms.ToTensor()(augmentation(img_pil)))
        augmented_images_2.append(transforms.ToTensor()(augmentation(img_pil)))
    
    train_images = torch.cat([train_images, torch.stack(augmented_images_1), torch.stack(augmented_images_2)], dim=0)
    train_labels = torch.cat([train_labels] * 3, dim=0)  # Triple labels to match augmented images
    
    print(f"Original training samples: 60000")
    print(f"After augmentation: {train_images.shape[0]}")
    
    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    model = train_model(train_loader, test_loader, epochs=5, learning_rate=0.001)
    
    print("\nTraining complete! Model saved and ready for use.")


if __name__ == '__main__':
    main()
