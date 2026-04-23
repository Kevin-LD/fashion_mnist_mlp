import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """
    解析 Fashion-MNIST 原始二进制文件
    kind: 'train' 或 't10k'
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, num = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows * cols)

    return images, labels

def prepare_data(data_dir, val_ratio=0.1):
    """
    加载数据并划分训练集与验证集
    """
    # 加载训练集和测试集
    X_train_full, y_train_full = load_mnist(data_dir, kind='train')
    X_test, y_test = load_mnist(data_dir, kind='t10k')

    # 划分验证集
    num_val = int(len(X_train_full) * val_ratio)
    indices = np.random.permutation(len(X_train_full))
    
    val_idx, train_idx = indices[:num_val], indices[num_val:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        """
        数据加载器
        X: (N, H*W) 的 NumPy 数组
        y: (N,) 的标签数组
        """
        self.X = X.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size # n_samples/batch_size 向上取整
