# Fashion-MNIST MLP 训练与评估
本项目基于三层多层感知机（MLP）对 Fashion-MNIST 数据集进行分类，所有组件均手动实现，不依赖 PyTorch 等深度学习框架。提供训练脚本 `train.py` 和评估脚本 `evaluate.py`。
## 环境依赖
请使用 Python 3.12 运行。安装所需依赖：
`pip install -r requirements.txt`
主要依赖为 numpy 等科学计算与下载工具库，详见 `requirements.txt`。
## 数据集
Fashion-MNIST 数据集将由脚本自动下载至 `--data_path` 指定的目录（默认为 `./data`），无需手动准备。
## 训练
运行 `train.py` 开始训练，基本命令：
`python train.py`
- ` --data_path`：数据存放地址（默认`./data`）。
- `--save_dir`：结果保存地址（默认``./runs/exp_时间戳``）。
- `--hidden1`：第一个隐藏层的神经元数量（默认256）。
- `--hidden2`：第二个隐藏层的神经元数量（默认128）。
- `--activation`：激活函数，可选 `relu`、`sigmoid`、`tanh`（默认 `relu`）。
- `--dropout_p`：Dropout 丢弃概率，0.0 表示不使用（默认0.0）。
- `--epochs`：训练总轮数（默认30）。
- `--batch_size`：训练批次大小（默认64）。
- `--val_batch_size`：验证批次大小（默认128）。
- `--lr`：初始学习率（默认0.1）；若续训并使用非线性调度器，此值作为基准。
- `--momentum`：SGD 动量（默认0.9）。
- `--weight_decay`：L2 正则化系数（默认1e-4）。
- `--scheduler`：学习率衰减策略，可选 `constant`、`step`、`linear`、`cosine`（默认 `constant`，即固定学习率）。
- `--step_size`：若使用 `step` 调度器，该参数指定衰减步长（每隔多少 epoch 衰减一次，默认10）。
- `--gamma`：若使用 `step` 调度器，该参数指定衰减率（默认0.1）。
- `--resume`：断点续训的权重文件路径（`.pkl`）。留空则不续训，从头开始（默认为空）。
- `--no_save_history`：添加此标志则不保存训练历史记录（`history.json` 及相关元数据）。
- `--seed`：全局随机种子，确保数据集切分一致（默认42）。
查看完整说明：`python train.py --help`
训练结束后，模型权重及训练记录将保存在 `--save_dir` 目录下（如未使用 `--no_save_history`）。
## 评估
使用 `evaluate.py` 计算模型在测试集上的准确率。必需参数：
- `--model_path`：待评估的权重文件路径（`.pkl`）。
基本命令：
`python evaluate.py --model_path ./runs/my_exp/best_model.pkl`
其他可选参数：
- `--data_path`：数据集目录（默认 `./data`）。
- `--batch_size`：评估批次大小（默认128）。
- `--hidden1`、`--hidden2`、`--activation`：当权重文件目录下不包含模型结构元数据时，需要通过这些参数显式指定网络结构，务必与训练时的设置一致（默认值分别为256、128、`relu`）；若权重文件目录下已保存结构信息，则无需指定。
查看完整说明：`python evaluate.py --help`
## 完整示例
（1）训练：两个隐藏层大小分别为512和256，使用ReLU，训练40个epoch，结果保存在 `./exp1`：
`python train.py --hidden1 512 --hidden2 256 --activation "relu" --epochs 40 --save_dir "./exp1"`  
（2）评估：
`python evaluate.py --model_path "./exp1/best_model.pkl"`
