import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from core.grad_mode import no_grad

# 确保导入项目模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datasets.dataset import prepare_data, DataLoader
from models.mlp import ThreeLayerMLP

def evaluate(model, data_loader, criterion=None, return_preds=False, desc="Evaluating"):
    """
    在给定数据集上评估模型性能的通用函数
    """
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = [] if return_preds else None
    all_targets = [] if return_preds else None
    
    pbar = tqdm(data_loader, desc=desc, leave=False)

    # 记录模型状态
    was_training = getattr(model, 'training', True)

    # 开启评估模式 (禁用 drop_out)
    model.eval()
    
    for X_batch, y_batch in pbar:
        # 禁用梯度
        with no_grad():
            logits = model.forward(X_batch)

            # 如果传入了损失函数，则计算 loss（主要用于 train.py 的验证环节）
            if criterion is not None:
                loss = criterion.forward(logits, y_batch)
                total_loss += loss * X_batch.shape[0]
        
            preds = np.argmax(logits, axis=1)
        
        correct += np.sum(preds == y_batch)
        total += X_batch.shape[0]
        
        # 如果需要返回预测结果用于混淆矩阵
        if return_preds:
            all_preds.extend(preds)
            all_targets.extend(y_batch)
            
    pbar.close()
            
    avg_loss = (total_loss / total) if criterion is not None else None
    accuracy = correct / total

    # 恢复模型状态
    if was_training:
        model.train()
    
    if return_preds:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)
    return avg_loss, accuracy

def run_evaluation(args):
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
        print(f"错误: 找不到权重文件 {args.model_path}")
        return
    model.load_weights(args.model_path)

    # 3. 加载测试集
    print("[*] 正在加载测试数据...")
    _, _, (X_test, y_test) = prepare_data(args.data_path)
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # 4. 推理
    print(f"[*] 开始评估...")
    _, accuracy = evaluate(model, test_loader, criterion=None, return_preds=False, desc="Testing")

    print("\n" + "="*30)
    print(f"测试准确率: {accuracy*100:.2f}%")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型准确率评估脚本")
    parser.add_argument('--model_path', type=str, required=True, help='权重文件路径 (.pkl)')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # 备选参数 (如果没有 meta 文件时起作用)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    run_evaluation(args)
