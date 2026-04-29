import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

def load_meta_data(model_path):
    """
    加载模型关联的元数据
    """
    meta_path = model_path.replace('.pkl', '_meta.json')
    if not os.path.exists(meta_path):
        print(f"Warning: Metadata file not found at {meta_path}")
        return None
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta

def extract_weights(model_path):
    """
    从 .pkl 二进制文件中提取第一层隐藏层的权重矩阵 (fc1_W)
    """
    try:
        with open(model_path, 'rb') as f:
            weights_dict = pickle.load(f)
        
        # 根据你定义的模型，第一层权重键名为 'fc1_W'
        if 'fc1_W' in weights_dict:
            w = weights_dict['fc1_W']
            print(f"Successfully extracted fc1_W. Shape: {w.shape}")
            return w
        else:
            print("Error: 'fc1_W' not found in the weight file.")
            return None
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None

def plot_weight_visualization(weights, save_path=None, num_neurons=64, col_idx=None):
    """
    将权重矩阵可视化。若指定 col_idx 则显示单列，否则显示图像网格。
    """
    input_size, hidden_size = weights.shape
    
    # 自动推断图像尺寸 (例如 784 -> 28x28)
    img_dim = int(math.sqrt(input_size))
    if img_dim * img_dim != input_size:
        print(f"Warning: Input size {input_size} is not a perfect square.")

    # 情况 A: 指定了可视化的某一列
    if col_idx is not None:
        if col_idx < 0 or col_idx >= hidden_size:
            print(f"Error: Column index {col_idx} is out of range (0-{hidden_size-1}).")
            return

        w_vector = weights[:, col_idx]
        w_img = w_vector.reshape(img_dim, img_dim)
        vmax = np.abs(w_img).max()

        plt.figure(figsize=(6, 5))
        im = plt.imshow(w_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.title(f'Weight Visualization: fc1 Column {col_idx}')
        plt.colorbar(im)
        plt.axis('off')
    # 情况 B: 默认展示网格
    else:
        # 确定显示数量和网格大小
        num_to_show = min(num_neurons, hidden_size)
        grid_size = int(math.ceil(math.sqrt(num_to_show)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle('Weight Visualization: 1st Hidden Layer (fc1)', fontsize=16)

        im = None
        for i in range(grid_size * grid_size):
            ax = axes.flat[i]
            if i < num_to_show:
                w_vector = weights[:, i]
                w_img = w_vector.reshape(img_dim, img_dim)
                
                # 使用发散型色图 RdBu
                vmax = np.abs(w_img).max()
                im = ax.imshow(w_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax.axis('off')


    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def run_visualize(args):
    """
    可视化主逻辑
    """
    print(f"--- Visualizing Pattern for Model: {args.model_path} ---")
    
    # 1. 加载元数据查看配置
    meta = load_meta_data(args.model_path)
    if meta:
        print(f"Metadata Info: {meta}")

    # 2. 提取权重数据
    weights = extract_weights(args.model_path)
    if weights is None:
        return

    # 3. 执行绘图
    model_dir = os.path.dirname(args.model_path)
    save_dir = args.save_dir if args.save_dir else os.path.join(model_dir, 'exp_figures')
    
    # 如果指定了列，修改保存文件名
    if args.column is not None:
        save_name = f'fc1_weights_col_{args.column}.png'
    else:
        save_name = 'fc1_weights_pattern.png'
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, save_name)
    
    plot_weight_visualization(weights, save_path=save_path, col_idx=args.column)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="手工实现 MLP 权重可视化脚本")
    
    # 核心路径参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的 .pkl 权重文件路径')
    parser.add_argument('--save_dir', type=str, default='', help='可视化结果保存目录')
    
    # 新增参数：指定可视化的某一列
    parser.add_argument('--column', type=int, default=None, help='指定可视化权重的某一列 (Neuron Index)')

    # 备选参数 (如果没有 meta 文件时起作用)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    run_visualize(args)
