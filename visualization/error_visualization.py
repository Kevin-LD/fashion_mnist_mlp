import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 确保可以导入根目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.dataset import prepare_data, DataLoader
from models.mlp import ThreeLayerMLP
from evaluate import evaluate

def plot_errors(images, preds, targets, class_names, save_path, num_samples=10):
    """
    绘制分类错误的样本图像
    """
    num_to_plot = min(len(images), num_samples)
    cols = 5
    rows = (num_to_plot + cols - 1) // cols
    
    plt.figure(figsize=(15, 3 * rows))
    
    for i in range(num_to_plot):
        plt.subplot(rows, cols, i + 1)
        # 将 784 维向量恢复为 28x28 图像
        img = images[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        pred_label = class_names[preds[i]]
        true_label = class_names[targets[i]]
        
        plt.title(f"Pred: {pred_label}\nActual: {true_label}", color='red', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] 错例分析图像已保存至: {save_path}")

def run_visualize(args):
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # 1. 尝试从元数据恢复配置
    meta_path = args.model_path.replace('.pkl', '_meta.json')
    activation = args.activation
    h1, h2 = args.hidden1, args.hidden2

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            config = meta.get('config', {})
            h1, h2 = config.get('hidden1', h1), config.get('hidden2', h2)
            activation = config.get('activation', activation)
            print(f"[*] 已从元数据加载配置: hidden1={h1}, hidden2={h2}, activation={activation}")

    # 2. 模型加载
    model = ThreeLayerMLP(input_dim=784, hidden_dim1=h1, hidden_dim2=h2, 
                          num_classes=10, activation=activation)
    model.load_weights(args.model_path)

    # 3. 加载数据 (需要 X_test 来显示图像)
    _, _, (X_test, y_test) = prepare_data(args.data_path)
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # 4. 获取预测结果
    print("[*] 正在运行测试集评估以寻找错误样本...")
    # 使用你提供的 evaluate 函数，返回预测值和真实值
    _, accuracy, all_preds, all_targets = evaluate(model, test_loader, return_preds=True)

    # 5. 筛选错误样本
    # 找出预测值与真实值不符的索引
    error_indices = np.where(all_preds != all_targets)[0]
    print(f"[*] 总测试样本: {len(all_targets)}, 错误数: {len(error_indices)}, 准确率: {accuracy*100:.2f}%")

    if len(error_indices) == 0:
        print("[!] 竟然一个都没分错？无法进行错例分析。")
        return

    # 提取错误的图像、预测结果和真实标签
    # 注意：X_test 的顺序必须与 evaluate 得到的 all_preds 顺序一致（所以 DataLoader shuffle=False）
    error_images = X_test[error_indices]
    error_preds = all_preds[error_indices]
    error_targets = all_targets[error_indices]

    # 6. 可视化并保存
    model_dir = os.path.dirname(args.model_path)
    save_dir = args.save_dir if args.save_dir else os.path.join(model_dir, 'exp_figures')
    save_path = os.path.join(save_dir, "error_analysis.png")
    
    plot_errors(error_images, error_preds, error_targets, class_names, save_path, num_samples=15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分类错例分析可视化脚本")
    parser.add_argument('--model_path', type=str, required=True, help='模型 .pkl 文件路径')
    parser.add_argument('--data_path', type=str, default='./data', help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='', help='图片保存目录')
    
    # 备选参数 (如果没有 meta 文件时起作用)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    run_visualize(args)
