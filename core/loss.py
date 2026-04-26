import numpy as np
from core.grad_mode import is_grad_enabled

class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def forward(self, logits, y_true):
        """
        前向传播
        :param logits: 最后一层线性层的输出 (batch_size, num_classes)
        :param y_true: 真实标签，形状为 (batch_size,) 的整数数组
        :return: 平均交叉熵损失
        """
        batch_size = logits.shape[0]
        
        # 1. Softmax 实现
        # 减去每行的最大值，防止 exp(logits) 溢出
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # 仅在需要求导时缓存预测概率和真实标签
        if is_grad_enabled():
            self.cache = (probs, y_true)
        
        # 2. 计算交叉熵损失: L = -sum(y_true * log(probs)) / batch_size
        # 添加 1e-12 防止 log(0)
        core_probs = probs[np.arange(batch_size), y_true]
        loss = -np.mean(np.log(core_probs + 1e-12))
        
        return loss

    def backward(self):
        """
        反向传播
        :return: 梯度 dZ，形状为 (batch_size, num_classes)
        """
        assert self.cache is not None, "Cannot call backward without forward cache."
        probs, y_true = self.cache
        batch_size = probs.shape[0]
        
        # 复制预测概率
        dZ = probs.copy()
        
        # 计算梯度: dL/dZ = (probs - y_true) / batch_size
        # 对于每个样本，在其真实类别所在的维度减去 1
        dZ[np.arange(batch_size), y_true] -= 1
        dZ = dZ / batch_size
        self.cache = None
        
        return dZ
