# Fashion-MNIST MLP 分类器
基于三层多层感知机（MLP）对 Fashion-MNIST 数据集进行图像分类，支持灵活配置网络结构、训练策略与断点续训。
## 环境依赖
- Python 3.8 或更高版本
- 依赖包见 `requirements.txt`，使用以下命令安装：`pip install -r requirements.txt`
主要依赖包括 PyTorch、torchvision、NumPy、Matplotlib、Seaborn、Pandas、tqdm 等。
## 数据准备
Fashion-MNIST 数据集会在首次运行时自动下载至 `--data_path` 指定的目录（默认为 `./data`），无需手动准备。
## 训练
运行 `train.py` 启动训练，基本命令为：
`python train.py --data_path ./data --save_dir ./results --epochs 50 --batch_size 64 --lr 0.01 --hidden1 256 --hidden2 128 --activation relu --dropout_p 0.3 --scheduler step --step_size 15 --gamma 0.5`
重要参数说明：
- `--data_path`：数据集存放/下载路径，默认 `./data`
- `--save_dir`：模型与历史记录保存目录，默认 `./results`
- `--epochs`：训练轮数，默认 `50`
- `--batch_size`：训练批次大小，默认 `64`
- `--lr`：初始学习率，默认 `0.01`
- `--momentum`：SGD 动量，默认 `0.9`
- `--weight_decay`：L2 正则化系数，默认 `0.0001`
- `--hidden1`：第一个隐藏层神经元数，默认 `256`
- `--hidden2`：第二个隐藏层神经元数，默认 `128`
- `--activation`：激活函数，可选 `relu`、`sigmoid`、`tanh`，默认 `relu`
- `--dropout_p`：Dropout 丢弃概率，`0.0` 表示不使用，默认 `0.0`
- `--scheduler`：学习率调度器，可选 `constant`、`step`、`linear`、`cosine`，默认 `step`
- `--step_size`：StepLR 的衰减步长（仅 step 调度器有效），默认 `15`
- `--gamma`：StepLR 的衰减率，默认 `0.5`
- `--seed`：全局随机种子，默认 `42`
- `--no_save_history`：添加此选项则不保存训练历史记录
- `--resume`：断点续训，指定已有权重文件路径（`.pkl`）
训练完成后，在 `save_dir` 目录下会生成：
- `model.pkl`：模型权重
- `history.json`：训练损失与准确率记录（除非使用 `--no_save_history`）
- 训练曲线图等（如未禁用历史记录）
## 评估
使用 `evaluate.py` 对保存的模型在测试集上计算准确率：
`python evaluate.py --model_path ./results/model.pkl --data_path ./data --hidden1 256 --hidden2 128 --activation relu`
**注意**：`--hidden1`、`--hidden2` 和 `--activation` 必须与训练时使用的网络结构完全一致，否则无法正确加载权重。
参数说明：
- `--model_path`：待评估的权重文件路径（.pkl），**必需**
- `--data_path`：数据集路径，默认 `./data`
- `--batch_size`：评估批次大小，默认 `256`
- `--hidden1`：与训练一致的隐藏层1大小，默认 `256`
- `--hidden2`：与训练一致的隐藏层2大小，默认 `128`
- `--activation`：与训练一致的激活函数，默认 `relu`
## 断点续训
若训练意外中断，可通过 `--resume` 继续训练：
`python train.py --resume ./results/model.pkl --epochs 100 --lr 0.001`
续训时学习率调度器会从断点前的状态继续，`--lr` 指定的值会被用作基准（仅在非线性调度时有效），权重与优化器状态完整恢复。
## 项目文件结构
`train.py` - 训练脚本 | `evaluate.py` - 评估脚本 | `requirements.txt` - 依赖 | `data/` - 数据集（自动下载）
