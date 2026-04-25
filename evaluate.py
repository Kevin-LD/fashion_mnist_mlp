import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 确保导入项目模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datasets.dataset import prepare_data, DataLoader
from models.mlp import ThreeLayerMLP

def calculate_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(args):
    # 类别定义
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
            activation = config.get('activation', activation)
            h1 = config.get('hidden1', h1)
            h2 = config.get('hidden2', h2)
            print(f"[*] 已从元数据自动恢复架构: {h1}-{h2} | 激活: {activation}")

    # 2. 初始化模型并加载权重
    model = ThreeLayerMLP(input_dim=784, hidden_dim1=h1, hidden_dim2=h2, 
                          num_classes=10, activation=activation)
    
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到文件 {args.model_path}")
        return
    model.load_weights(args.model_path)

    # 3. 加载测试集
    print("[*] 正在加载测试数据...")
    _, _, (X_test, y_test) = prepare_data(args.data_path)
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # 4. 推理
    print(f"[*] 开始评估...")
    all_preds = []
    for X_batch, _ in test_loader:
        logits = model.forward(X_batch)
        all_preds.extend(np.argmax(logits, axis=1))
    
    all_preds = np.array(all_preds)
    accuracy = np.mean(all_preds == y_test)

    print("\n" + "="*30)
    print(f"测试准确率: {accuracy*100:.2f}%")
    print("="*30)

    # 5. 可视化混淆矩阵
    cm = calculate_confusion_matrix(y_test, all_preds)
    
    # 自动确定图片保存位置
    model_dir = os.path.dirname(args.model_path)
    save_dir = args.save_dir if args.save_dir else os.path.join(model_dir, 'exp_figures')
    save_path = os.path.join(save_dir, f"cm_acc_{accuracy*100:.1f}.png")
    
    plot_confusion_matrix(cm, class_names, save_path)
    print(f"[*] 混淆矩阵已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='权重文件路径 (.pkl)')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='', help='图片保存目录')
    
    # 备选参数 (如果没有 meta 文件时起作用)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    evaluate_model(args)
