import struct
import numpy as np


def read_idx_ubyte(filename, asbytes=False):
    """
    Read idx ubyte file format (MNIST dataset format)
    """
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic == 2051:  # images
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        elif magic == 2049:  # labels
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError(f"Invalid magic number: {magic}")
    
    return data if asbytes else data.astype(np.float32) / 255.0


def load_mnist_data(data_dir):
    """
    Load MNIST training and test data
    """
    print("Loading MNIST dataset...")
    
    train_images = read_idx_ubyte(f'{data_dir}/train-images.idx3-ubyte')
    train_labels = read_idx_ubyte(f'{data_dir}/train-labels.idx1-ubyte', asbytes=True)
    test_images = read_idx_ubyte(f'{data_dir}/t10k-images.idx3-ubyte')
    test_labels = read_idx_ubyte(f'{data_dir}/t10k-labels.idx1-ubyte', asbytes=True)
    
    print(f"Training set: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Test set: {test_images.shape} images, {test_labels.shape} labels")
    
    return train_images, train_labels, test_images, test_labels
