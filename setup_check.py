"""
Quick setup verification script
Run this to verify everything is installed correctly before training
"""

import sys

def check_packages():
    """Check if all required packages are installed"""
    packages = {
        'torch': 'PyTorch',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
    }
    
    print("🔍 Checking installed packages...\n")
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nRun this to install missing packages:")
        print(f"pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All packages installed correctly!")
        return True


def check_data_files():
    """Check if MNIST data files exist"""
    import os
    
    print("\n🔍 Checking data files...\n")
    
    required_files = [
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte',
    ]
    
    missing = []
    for filename in required_files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"✅ {filename} - {size_mb:.1f} MB")
        else:
            print(f"❌ {filename} - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  Missing data files: {', '.join(missing)}")
        print(f"Make sure MNIST dataset files are in the current directory")
        return False
    else:
        print(f"\n✅ All data files found!")
        return True


def main():
    print("=" * 50)
    print("  🔢 Handwriting Digit Detector - Setup Check")
    print("=" * 50)
    
    packages_ok = check_packages()
    data_ok = check_data_files()
    
    print("\n" + "=" * 50)
    if packages_ok and data_ok:
        print("✅ Setup is complete! You can now:")
        print("   1. Run: python train_model.py")
        print("   2. Then: streamlit run app.py")
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == '__main__':
    main()
