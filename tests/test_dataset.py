import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 将根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.dataset import prepare_data, DataLoader

def test_dataset_loading():
    print("正在加载数据并进行初步检查...")
    data_path = '../data'
    
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(data_path)
    except FileNotFoundError:
        print(f"错误：请确保在 {data_path} 目录下存放了 Fashion-MNIST 的二进制文件。")
        return

    # 1. 检查数据维度
    print(f"训练集尺寸: {X_train.shape}, 标签尺寸: {y_train.shape}")
    print(f"验证集尺寸: {X_val.shape}")
    print(f"测试集尺寸: {X_test.shape}")
    
    assert X_train.shape[1] == 784, "图像维度应为 784 (28x28)"
    assert len(np.unique(y_train)) == 10, "类别数量应为 10"

    # 2. 检查 DataLoader 与归一化
    batch_size = 64
    loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    
    X_batch, y_batch = next(iter(loader))
    
    print("\nDataLoader 检查:")
    print(f"Batch X 形状: {X_batch.shape}")
    print(f"Batch y 形状: {y_batch.shape}")
    print(f"数值范围: [{X_batch.min()}, {X_batch.max()}]")
    
    assert X_batch.shape == (batch_size, 784), "Batch 形状不正确"
    assert X_batch.max() <= 1.0 and X_batch.min() >= 0.0, "数据未正确归一化到 [0, 1]"

    # 3. 可视化检查
    print("\n正在生成样本预览...")
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
        plt.title(classes[y_batch[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dataset_loading()
