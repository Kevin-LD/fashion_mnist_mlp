import os
import json
import random
import math
import itertools
from datetime import datetime
from argparse import Namespace

from train import train

def get_base_args():
    """获取基础参数配置 (固定不变的参数)"""
    return Namespace(
        data_path='./data',
        resume='',
        epochs=2,             # 在搜索时适当减少 epochs 以加快搜索速度
        val_batch_size=128,
        momentum=0.9,
        scheduler='cosine',    # 搜索时固定使用一种表现较好的调度器
        step_size=10,
        gamma=0.1,
        no_save_history=True   # [!] 搜索时不保存历史记录，节省磁盘空间
    )

def run_search(search_type='random', num_trials=10):
    """
    执行超参数搜索
    :param search_type: 'grid' (网格搜索) 或 'random' (随机搜索)
    :param num_trials: 随机搜索的尝试次数 (网格搜索忽略此参数)
    """
    # 1. 定义搜索空间
    # 对于 grid: 使用 'grid_values' 中的列表
    # 对于 random: 如果有 'range' 则在范围内采样，否则从 'grid_values' 中随机选
    search_space = {
        'lr': {
            'range': [1e-5, 1e-1], 
            'scale': 'log', 
            'grid_values': [1e-2, 1e-3, 1e-4]
        },
        'hidden1': {
            'grid_values': [256, 512]
        },
        'hidden2': {
            'grid_values': [64, 128]
        },
        'weight_decay': {
            'range': [1e-6, 1e-3], 
            'scale': 'log', 
            'grid_values': [1e-4, 1e-5]
        },
        'activation': {
            'grid_values': ['relu', 'tanh']
        },
        'batch_size': {
            'grid_values': [32, 64, 128] # 习惯上保持 2 的倍数，故使用离散值
        }
    }

    keys = list(search_space.keys())
    combinations = []

    # 2. 生成超参数组合列表
    if search_type == 'grid':
        # 网格搜索：仅使用 grid_values 进行笛卡尔积
        grid_lists = [search_space[k]['grid_values'] for k in keys]
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*grid_lists)]
        print(f"[*] 启动网格搜索，共计 {len(combinations)} 组参数组合。")
    else:
        # 随机搜索：根据配置进行连续或离散采样
        print(f"[*] 启动随机搜索，将尝试 {num_trials} 组随机参数组合。")
        for _ in range(num_trials):
            combo = {}
            for k, config in search_space.items():
                if 'range' in config:
                    # 连续空间采样
                    low, high = config['range']
                    if config.get('scale') == 'log':
                        # 对数尺度随机采样: 10^uniform(log10(low), log10(high))
                        val = 10 ** random.uniform(math.log10(low), math.log10(high))
                    else:
                        # 线性尺度随机采样
                        val = random.uniform(low, high)
                    combo[k] = val
                else:
                    # 离散空间采样
                    combo[k] = random.choice(config['grid_values'])
            combinations.append(combo)

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_dir = f'./runs/search_{search_type}_{timestamp}'
    os.makedirs(search_dir, exist_ok=True)

    # 3. 开始遍历超参数组合
    for i, params in enumerate(combinations):
        print(f"\n" + "="*50)
        print(f"Trial [{i+1}/{len(combinations)}] - 当前参数: {params}")
        print("="*50)

        # 构造当前 Trial 的 Args
        args = get_base_args()
        for k, v in params.items():
            setattr(args, k, v) # 将动态参数注入 args
            
        # 为每个 trial 设置独立的数据保存路径 (对浮点数进行格式化防止文件名过长)
        trial_name = f"trial_{i+1}_lr{args.lr:.2e}_h1{args.hidden1}_bs{args.batch_size}"
        args.save_dir = os.path.join(search_dir, "trials", trial_name)

        # 4. 执行训练并捕获异常
        try:
            best_val_acc, history = train(args)
            
            trial_result = {
                'trial': i + 1,
                'params': params,
                'best_val_acc': best_val_acc,
                'save_dir': args.save_dir
            }
            results.append(trial_result)
            
        except Exception as e:
            print(f"[!] Trial {i+1} 训练失败: {e}")
            results.append({
                'trial': i + 1,
                'params': params,
                'best_val_acc': 0.0,
                'error': str(e)
            })

    # 5. 汇总并输出结果
    print("\n" + "="*60)
    print("搜索完成！(Search Completed)")
    print("="*60)
    
    results = sorted(results, key=lambda x: x.get('best_val_acc', 0.0), reverse=True)
    
    print("\nTop 3 参数组合:")
    for i in range(min(3, len(results))):
        res = results[i]
        print(f"Top {i+1} | Acc: {res['best_val_acc']*100:.2f}% | Params: {res['params']}")

    summary_path = os.path.join(search_dir, 'search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n完整的搜索报告已保存至: {summary_path}")

if __name__ == '__main__':
    # 随机搜索
    run_search(search_type='random', num_trials=3)
    # 网格搜索
    # run_search(search_type='grid')
