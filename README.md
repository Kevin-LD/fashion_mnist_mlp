# Fashion-MNIST MLP
## 项目简介  
本项目基于 NumPy 从零实现了一个三层多层感知机（MLP）分类器，用于 Fashion-MNIST 数据集的图像分类任务。项目完整实现了前向传播、反向传播、损失计算、优化器、学习率调度与训练流程，并支持隐藏层结构、激活函数、正则化等超参数配置。通过系统化的超参数搜索，最终模型在测试集上取得约 90.53% 的分类准确率。 
## 环境依赖  
- Python 3.12  
- 依赖库见 `requirements.txt`（使用 `pip install -r requirements.txt` 安装）  
## 数据准备  
首次运行训练脚本时，程序会自动下载并准备 Fashion-MNIST 数据集，无需手动下载。数据默认保存在 `./data` 目录下。  
## 运行方式  
### 训练  
使用 train.py 进行模型训练，支持学习率调度、Dropout 与断点续训等功能。 
示例：
```bash
python train.py --hidden1 512 --hidden2 128 --weight_decay 1e-4 --batch_size 32 --lr 2e-2 --scheduler "cosine" --epochs 200 --dropout_p 0.2 --save_dir "runs/exp_2_epoch_200"         
``` 
说明：  
- `--hidden1 / --hidden2`：隐藏层维度  
- `--weight_decay`：L2 正则化强度  
- `--batch_size`：Mini-batch 大小  
- `--lr`：学习率  
- `--scheduler`：学习率调度策略（constant / step / linear / cosine）  
- `--epochs`：训练轮数  
- `--dropout_p`：Dropout 概率  
- `--save_dir`：模型权重与训练日志保存目录  
### 测试  
使用 `evaluate.py` 对训练好的模型进行评估。  
示例：  
```bash
python evaluate.py --model_path "runs\exp_2_epoch_200\best_model.pkl"
```
测试脚本将输出测试集上的分类准确率。  
### 可视化
使用 viasualization 目录下的脚本对训练结果进行可视化。
示例：
```bash
python visualization\\confusion_matrix_visualization.py --model_path "runs\exp_2_epoch_200\best_model.pkl"                                                                     
```
训练、测试与可视化脚本的详细参数说明可通过以下命令查看：
```bash
python script_name.py --help
```
## 模型权重  
训练完成后的模型权重可通过以下链接下载：  
https://drive.google.com/file/d/1oR0kwrEkUiJceKnIKqiBwuYCAAms9fPR/view?usp=sharing
