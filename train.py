import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

# 确保能正确导入当前项目的模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datasets.dataset import prepare_data, DataLoader
from models.mlp import ThreeLayerMLP
from core.loss import CrossEntropyLoss
from core.optimizer import SGD
from core.scheduler import ConstantLR, StepLR, LinearLR, CosineAnnealingLR

from evaluate import evaluate 

def train(args):
    print("="*60)
    print(f"实验配置:")
    print(f"学习率: {args.lr} | 权重衰减: {args.weight_decay} | 调度器: {args.scheduler}")
    print(f"批次大小: 训练集 {args.batch_size}, 验证集 {args.val_batch_size}")
    print(f"隐藏层维度: [{args.hidden1}, {args.hidden2}] | 激活函数: {args.activation}")
    print(f"Dropout 概率: {args.dropout_p} | 保存目录: {args.save_dir}")
    print("="*60)

    # 1. 准备数据
    try:
        (X_train, y_train), (X_val, y_val), _ = prepare_data(args.data_path, val_ratio=0.1)
    except FileNotFoundError:
        print(f"错误: 找不到数据集，请确认 {args.data_path} 下存在二进制文件")
        return

    train_loader = DataLoader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=args.val_batch_size, shuffle=False)

    # 2. 初始化模型、损失函数
    model = ThreeLayerMLP(
        input_dim=784,
        hidden_dim1=args.hidden1,
        hidden_dim2=args.hidden2,
        num_classes=10,
        weight_decay=args.weight_decay,
        activation=args.activation,
        dropout_p=args.dropout_p
    )
    criterion = CrossEntropyLoss()

    # 3. 初始化状态与全局步数计算
    start_epoch = 0
    last_step_meta = -1 # 默认从头开始
    best_val_acc_from_meta = 0.0 # 用于在无 history 时恢复最佳准确率
    
    history = {
        'epoch': {'train_loss': [], 'train_data_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []},
        'iter': {'train_loss': [], 'train_data_loss': [], 'lr': []}
    }

    # 计算当前 batch_size 下的总 iter 步数
    total_iters_per_epoch = len(train_loader.X) // args.batch_size + (1 if len(train_loader.X) % args.batch_size != 0 else 0)
    total_steps = args.epochs * total_iters_per_epoch

    # 4. 断点续训
    meta_path = ""
    if args.resume and os.path.exists(args.resume):
        print(f"正在从断点恢复模型: {args.resume}")
        model.load_weights(args.resume)
        
        resume_dir = os.path.dirname(args.resume)
        meta_path = args.resume.replace('.pkl', '_meta.json')
        history_path = os.path.join(resume_dir, 'history.json')
        
        # 4.1 加载模型元数据
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                start_epoch = meta.get('epoch', 0)
                old_batch_size = meta.get('batch_size', args.batch_size)
                best_val_acc_from_meta = meta.get('best_val_acc', 0.0)
                saved_dropout = meta.get('config', {}).get('dropout_p', 0.0)

                # 处理 Dropout 断点续训逻辑
                user_provided_dropout = '--dropout_p' in sys.argv
                
                if user_provided_dropout:
                    # 如果用户显式输入了 dropout_p，听用户的，并给出警告
                    if saved_dropout != args.dropout_p:
                        print(f"[*] 警告: 断点配置(p={saved_dropout}) 与当前输入的参数(p={args.dropout_p}) 不一致！")
                        print(f"[*] 将遵从用户输入，使用当前的 Dropout 概率 {args.dropout_p}。")
                        model.drop1.p = args.dropout_p
                        model.drop2.p = args.dropout_p
                else:
                    # 如果用户没输入，听 meta 当中的，以保证连续性
                    if saved_dropout != args.dropout_p:
                        print(f"[*] 提示: 检测到默认参数(p={args.dropout_p}) 与断点配置(p={saved_dropout}) 不一致。")
                        print(f"[*] 用户未显式指定，正在自动恢复使用断点中的 Dropout 概率 {saved_dropout} 以保证训练连续性。")
                        args.dropout_p = saved_dropout
                        model.drop1.p = saved_dropout
                        model.drop2.p = saved_dropout
                
                if old_batch_size != args.batch_size:
                    print(f"[*] 检测到 batch_size 改变 ({old_batch_size} -> {args.batch_size})，正在自适应调整调度器步数...")
                
                if args.scheduler == 'step':
                    last_step_meta = start_epoch - 1
                else:
                    last_step_meta = (start_epoch * total_iters_per_epoch) - 1
                    
            print(f"元数据已加载: 模型权重来自 Epoch {start_epoch}。")
        else:
            print("警告: 未找到元数据文件 (_meta.json)。将使用默认调度器状态。")

        # 4.2 加载并截断历史记录 (对齐到 start_epoch)
        if not args.no_save_history and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                loaded_history = json.load(f)
                
            print(f"正在同步历史记录... (截断超出 Epoch {start_epoch} 的冗余数据)")
            
            # 截断 epoch 级别的记录
            for key in loaded_history['epoch']:
                if key in history['epoch']:
                    history['epoch'][key] = loaded_history['epoch'][key][:start_epoch]
                    
            # 截断 iter 级别的记录
            old_iters_per_epoch = len(train_loader.X) // old_batch_size + (1 if len(train_loader.X) % old_batch_size != 0 else 0)
            target_iters = start_epoch * old_iters_per_epoch
            for key in loaded_history['iter']:
                if key in history['iter']:
                    history['iter'][key] = loaded_history['iter'][key][:target_iters]
        elif not args.no_save_history:
            print("警告: 未找到 history.json 文件，将重新初始化历史记录记录。")

    # 检查训练进度是否已满
    if start_epoch >= args.epochs:
        print(f"\n[终止] 续训安全检查拦截: 当前已训练轮数 ({start_epoch}) 已达到或超过总目标轮数 ({args.epochs})。")
        print("请在启动时指定更大的 --epochs 以继续训练。")
        return

    # 5. 初始化优化器、调度器
    optimizer = SGD(model.get_trainable_layers(), lr=args.lr, momentum=args.momentum)
    
    if args.scheduler == 'constant':
        scheduler = ConstantLR(optimizer, last_step=last_step_meta)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_step=last_step_meta)
    elif args.scheduler == 'linear':
        scheduler = LinearLR(optimizer, total_steps=total_steps, last_step=last_step_meta)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, total_steps=total_steps, last_step=last_step_meta)

    # 6. 准备实验目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    best_model_path = os.path.join(args.save_dir, 'best_model.pkl')
    # 计算当前的最佳准确率，如果历史记录为空则使用 meta 中的记录
    best_val_acc = max(history['epoch']['val_acc']) if history['epoch']['val_acc'] else best_val_acc_from_meta

    # 7. 开始训练循环
    for epoch in range(start_epoch, args.epochs):
        # 开启训练模式 (启用 Dropout) 
        model.train()
        
        epoch_loss_sum = 0.0
        epoch_data_loss_sum = 0.0
        
        pbar = tqdm(train_loader, total=total_iters_per_epoch, leave=False)
        pbar.set_description(f"Epoch [{epoch+1:03d}/{args.epochs:03d}]")
        
        for X_batch, y_batch in pbar:
            logits = model.forward(X_batch)
            data_loss = criterion.forward(logits, y_batch)
            loss = data_loss + model.total_regularization_loss()
            
            model.backward(criterion.backward())
            optimizer.step()
            
            # 更新进度条右侧显示的实时信息
            pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{optimizer.lr:.6f}'})
            
            # 记录数据
            if not args.no_save_history:
                history['iter']['train_loss'].append(float(loss))
                history['iter']['train_data_loss'].append(float(data_loss))
                history['iter']['lr'].append(float(optimizer.lr))
                
            epoch_loss_sum += loss * X_batch.shape[0]
            epoch_data_loss_sum += data_loss * X_batch.shape[0]
            
            if args.scheduler in ['linear', 'cosine']:
                scheduler.step()
        
        pbar.close()
                
        if args.scheduler in ['step', 'constant']:
            scheduler.step()
            
        avg_train_loss = epoch_loss_sum / len(train_loader.X)
        avg_train_data_loss = epoch_data_loss_sum / len(train_loader.X)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # 记录数据
        if not args.no_save_history:
            history['epoch']['train_loss'].append(float(avg_train_loss))
            history['epoch']['train_data_loss'].append(float(avg_train_data_loss))
            history['epoch']['val_loss'].append(float(val_loss))
            history['epoch']['val_acc'].append(float(val_acc))
            history['epoch']['lr'].append(float(optimizer.lr))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(best_model_path)
            meta = {
                'epoch': epoch + 1,
                'last_step': scheduler.last_step,
                'batch_size': args.batch_size,
                'best_val_acc': best_val_acc,
                'config': {
                    'hidden1': args.hidden1,
                    'hidden2': args.hidden2,
                    'activation': args.activation,
                    'dropout_p': args.dropout_p
                }
            }
            with open(best_model_path.replace('.pkl', '_meta.json'), 'w') as f:
                json.dump(meta, f)
            best_flag = "★"
        else:
            best_flag = ""
            
        print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | "
              f"LR: {optimizer.lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% {best_flag}")

        # 统一在此处保存全局的历史记录
        if not args.no_save_history:
            with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
                json.dump(history, f)

    print("\n训练完成 (Training Complete).")
    return best_val_acc, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="用于 Fashion-MNIST 数据集的 3 层 MLP 训练脚本")
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_save_dir = os.path.join('./runs', f'exp_{current_time}')

    # 路径与检查点参数
    parser.add_argument('--data_path', type=str, default='./data', help='数据集所在目录路径')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='实验结果与模型保存路径')
    parser.add_argument('--resume', type=str, default='', help='用于断点续训的模型权重路径 (.pkl 文件)')
    
    # 网络架构参数
    parser.add_argument('--hidden1', type=int, default=256, help='第一个隐藏层的神经元数量')
    parser.add_argument('--hidden2', type=int, default=128, help='第二个隐藏层的神经元数量')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='网络使用的激活函数')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='Dropout 丢弃概率 (0.0 表示不使用)')
    
    # 优化与训练参数
    parser.add_argument('--epochs', type=int, default=30, help='训练的总轮数 (Epochs)')
    parser.add_argument('--batch_size', type=int, default=64, help='训练时的批次大小 (Batch Size)')
    parser.add_argument('--val_batch_size', type=int, default=128, help='验证时的批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率 (如果续训且非线性调度器，此值作为基准)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 优化器的动量 (Momentum)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减 (L2 正则化系数)')
    
    # 调度器参数
    parser.add_argument('--scheduler', type=str, default='constant', 
                        choices=['constant', 'step', 'linear', 'cosine'], help='学习率衰减策略')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR 的步长 (每隔多少个 epoch 衰减一次)')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR 的学习率衰减率')
    
    # 日志与保存控制
    parser.add_argument('--no_save_history', action='store_true', default=False, 
                        help='不保存训练历史记录 (history.json 及相关元数据)')
    
    args = parser.parse_args()
    train(args)
