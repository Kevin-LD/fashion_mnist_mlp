import os
import json
import random
import math
import itertools
import argparse
import copy
from datetime import datetime
from argparse import Namespace

from train import train

def get_base_args(cli_args):
    """获取基础参数配置 (固定不变的参数)"""
    return Namespace(
        data_path=cli_args.data_path,
        resume='',
        epochs=cli_args.epochs,        # 在搜索时适当减少 epochs 以加快搜索速度
        val_batch_size=128,
        momentum=0.9,
        scheduler='cosine',             # 搜索时固定使用一种表现较好的调度器
        step_size=10,
        gamma=0.1,
        no_save_history=True,           # [!] 搜索时不保存历史记录，节省磁盘空间
        # 已经搜索过的超参数
        activation='relu',
        lr=1e-2,
        batch_size=32
    )

def run_search(cli_args):
    """
    执行超参数搜索
    :param cli_args: 从命令行解析到的参数
    """
    search_type = cli_args.type
    num_trials = cli_args.trials

    # 基准配置 (用于缩放 lr)
    base_batch_size = 32

    # 1. 定义搜索空间
    search_space = {
        'weight_decay': {
            'range': [1e-6, 1e-3], 
            'scale': 'log', 
            'grid_values': [1e-4, 1e-5, 1e-6, 0]
        },
        'hidden1': {
            'grid_values': [256, 512]
        },
        'hidden2': {
            'grid_values': [64, 128]
        }
        # 已经搜索过的超参数
        # 'activation': {
        #     'grid_values': ['relu', 'tanh']
        # },
        # 'lr': {
        #     'range': [1e-5, 1e-1], 
        #     'scale': 'log', 
        #     'grid_values': [5e-2, 1e-2, 5e-3]
        # },
        # 'batch_size': {
        #     'grid_values': [32, 64]
        # },
    }

    keys = list(search_space.keys())
    combinations = []

    # 2. 生成超参数组合列表
    if search_type == 'grid':
        grid_lists = [search_space[k]['grid_values'] for k in keys]
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*grid_lists)]
        print(f"[*] 启动网格搜索，共计 {len(combinations)} 组参数组合。")
    else:
        print(f"[*] 启动随机搜索，将尝试 {num_trials} 组随机参数组合。")
        for _ in range(num_trials):
            combo = {}
            for k, config in search_space.items():
                if 'range' in config:
                    low, high = config['range']
                    if config.get('scale') == 'log':
                        val = 10 ** random.uniform(math.log10(low), math.log10(high))
                    else:
                        val = random.uniform(low, high)
                    combo[k] = val
                else:
                    combo[k] = random.choice(config['grid_values'])
            combinations.append(combo)

    # --- 路径处理逻辑 ---
    if cli_args.search_dir:
        search_dir = cli_args.search_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        search_dir = f'./runs/search_{search_type}_{timestamp}'
    
    os.makedirs(search_dir, exist_ok=True)
    print(f"[*] 搜索结果将保存至: {search_dir}")

    results = []

    # 3. 开始遍历超参数组合
    for i, params in enumerate(combinations):
        print(f"\n" + "="*50)
        print(f"Trial [{i+1}/{len(combinations)}] - 当前参数: {params}")
        print("="*50)

        args = get_base_args(cli_args)
        for k, v in params.items():
            setattr(args, k, v)

        # 根据 Linear Scaling Rule: lr 正比于 Batch size，epoch 保持一致
        scale = args.batch_size / base_batch_size

        # lr 正比于 batch size
        original_lr = args.lr
        args.lr = original_lr * scale

        print(f"[*] Batch Scaling: bs={args.batch_size}, scale={scale:.2f}")
        print(f"[*] Adjusted lr: {original_lr:.2e} -> {args.lr:.2e}")

        # 为每个 trial 设置独立路径
        trial_name = f"trial_{i+1}_lr{args.lr:.2e}_h1{args.hidden1}_bs{args.batch_size}"
        args.save_dir = os.path.join(search_dir, "trials", trial_name)

        # 4. 执行训练
        try:
            best_val_acc, _ = train(args)
            
            trial_result = {
                'trial': i + 1,
                'params': params,
                'full_params': copy.deepcopy(vars(args)),
                'best_val_acc': best_val_acc,
            }
            results.append(trial_result)
            
        except Exception as e:
            print(f"[!] Trial {i+1} 训练失败: {e}")
            results.append({
                'trial': i + 1,
                'params': params,
                'full_params': copy.deepcopy(vars(args)),
                'best_val_acc': 0.0,
                'error': str(e)
            })

# 5. 汇总结果
    print("\n" + "="*60)
    print("搜索完成！(Search Completed)")
    print("="*60)
    
    results = sorted(results, key=lambda x: x.get('best_val_acc', 0.0), reverse=True)
    
    print("\nTop 3 推荐配置:")
    for i in range(min(3, len(results))):
        res = results[i]
        p = res['params']
        fp = res['full_params']

        print(f"Top {i+1} | Acc: {res['best_val_acc']*100:.2f}%")

        print(f"  > 搜索参数: {p}")
        
        print(f"  [!] 最终训练建议：")
        print(f"  - 当前 lr 已根据 batch_size={fp['batch_size']} 进行线性缩放")
        print(f"  - 若保持 batch_size={fp['batch_size']}, 请直接使用 lr={fp['lr']:.2e}")
        print(f"  - 若更改 batch_size, 请按比例缩放: Final_LR = {fp['lr']:.2e} * (New_BS / {fp['batch_size']})")
        print("-" * 30)

    summary_path = os.path.join(search_dir, 'search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n完整的搜索报告已保存至: {summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="超参数搜索脚本")
    
    parser.add_argument('--type', type=str, default='random', choices=['grid', 'random'],
                        help='搜索模式: random (随机) 或 grid (网格)')
    parser.add_argument('--trials', type=int, default=30, 
                        help='随机搜索的尝试次数 (网格搜索下此参数失效)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='每个 trial (参数组合) 的 epoch 数量')
    parser.add_argument('--data_path', type=str, default='./data', 
                        help='数据集所在目录路径')
    parser.add_argument('--search_dir', type=str, default='', 
                        help='搜索结果保存的根目录 (不填则自动按时间戳生成)')

    args = parser.parse_args()

    run_search(args)
