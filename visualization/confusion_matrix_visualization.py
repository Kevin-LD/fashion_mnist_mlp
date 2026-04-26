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

def calculate_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    计算混淆矩阵
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, class_names, save_path):
    """
    绘制并保存混淆矩阵
    """
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
    print(f"[*] 混淆矩阵已保存至: {save_path}")

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

    # 2. 模型加载
    model = ThreeLayerMLP(input_dim=784, hidden_dim1=h1, hidden_dim2=h2, 
                          num_classes=10, activation=activation)
    model.load_weights(args.model_path)

    # 3. 加载数据
    _, _, (X_test, y_test) = prepare_data(args.data_path)
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # 4. 获取预测结果
    print("[*] 正在获取模型预测以生成混淆矩阵...")
    _, accuracy, all_preds, all_targets = evaluate(model, test_loader, return_preds=True)

    # 5. 计算与可视化
    cm = calculate_confusion_matrix(all_targets, all_preds)
    
    model_dir = os.path.dirname(args.model_path)
    save_dir = args.save_dir if args.save_dir else os.path.join(model_dir, 'exp_figures')
    save_path = os.path.join(save_dir, f"cm_acc_{accuracy*100:.1f}.png")
    
    plot_confusion_matrix(cm, class_names, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="混淆矩阵可视化脚本")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='', help='图片保存目录')
    
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    run_visualize(args)
