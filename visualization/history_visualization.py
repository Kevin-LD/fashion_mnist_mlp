import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.95):
    """
    用于平滑 iteration 级别的 loss 曲线。
    factor 越大 (接近1)，平滑力度越大。
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_history(exp_dir):
    # 1. 确定历史文件路径
    history_path = os.path.join(exp_dir, 'history.json')
    if not os.path.exists(history_path):
        print(f"找不到历史记录文件: {history_path}")
        return

    # 2. 读取 JSON 数据
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    # 提取数据
    epoch_train_loss = history['epoch']['train_loss']
    epoch_train_data_loss = history['epoch']['train_data_loss'] # 提取未加正则化的 data loss
    epoch_val_loss = history['epoch']['val_loss']
    epoch_val_acc = history['epoch']['val_acc']
    
    iter_train_loss = history['iter']['train_loss']
    iter_lr = history['iter']['lr']
    
    epochs = range(1, len(epoch_train_loss) + 1)
    iters = range(1, len(iter_train_loss) + 1)

    # 3. 创建画布 (2x2 布局)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History Visualization', fontsize=16)

    # --- 图 1: Epoch 级别的 Data Loss vs Val Loss ---
    axs[0, 0].plot(epochs, epoch_train_data_loss, 'b-', label='Train Data Loss', marker='o', markersize=4)
    axs[0, 0].plot(epochs, epoch_val_loss, 'r-', label='Val Loss', marker='s', markersize=4)
    axs[0, 0].set_title('Epoch Train Data Loss vs Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # --- 图 2: Epoch 级别的 Validation Accuracy ---
    axs[0, 1].plot(epochs, epoch_val_acc, 'g-', label='Val Accuracy', marker='^', markersize=4)
    axs[0, 1].set_title('Epoch Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # --- 图 3: Epoch 级别的 Total Loss ---
    axs[1, 0].plot(epochs, epoch_train_loss, 'purple', label='Train Total Loss', marker='d', markersize=4)
    axs[1, 0].set_title('Epoch Total Loss (Data + Regularization)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Total Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # --- 图 4: Iteration 级别的 Train Loss 与 LR 叠加显示 (双 Y 轴) ---
    color_loss = 'tab:blue'
    axs[1, 1].set_title('Iteration Loss & Learning Rate Schedule')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Loss', color=color_loss)
    
    # 画 Loss
    l1 = axs[1, 1].plot(iters, iter_train_loss, color='lightblue', alpha=0.3, label='Iter Loss (Raw)')
    l2 = axs[1, 1].plot(iters, smooth_curve(iter_train_loss, factor=0.95), color=color_loss, label='Iter Loss (Smoothed)')
    axs[1, 1].tick_params(axis='y', labelcolor=color_loss)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    # 实例化第二个 Y 轴，共享同一个 X 轴
    ax2 = axs[1, 1].twinx()
    color_lr = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color_lr)
    
    # 画 LR
    l3 = ax2.plot(iters, iter_lr, color=color_lr, linestyle='-', linewidth=2, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    
    # 如果 LR 变化超过几个数量级，可以取消下一行的注释使用对数坐标
    # ax2.set_yscale('log')

    # 合并双 Y 轴的图例
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    fig.subplots_adjust(top=0.9) # 给主标题留出空间

    # 4. 保存图表并显示
    save_dir = os.path.join(exp_dir, 'exp_figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'history_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"可视化图表已保存至: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(tools_dir)
    
    default_exp_dir = os.path.join(root_dir, 'runs', 'exp_01')

    parser = argparse.ArgumentParser(description='Visualize training history.')
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, 
                        help='Path to the experiment directory containing history.json')
    
    args = parser.parse_args()
    
    plot_history(args.exp_dir)
