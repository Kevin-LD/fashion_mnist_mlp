import numpy as np
from core.grad_mode import is_grad_enabled

class Linear:
    def __init__(self, in_features, out_features, weight_decay=0.0, init_method='he'):
        """
        全连接层初始化
        :param in_features: 输入特征维度 (n_in)
        :param out_features: 输出特征维度 (n_out)
        :param weight_decay: L2 正则化系数 (lambda)
        :param init_method: 'he', 'xavier', 或 'normal'
        """
        self.weight_decay = weight_decay
        
        # 初始化权重
        if init_method == 'he':
            # 适用于 ReLU: std = sqrt(2/n_in)
            limit = np.sqrt(2.0 / in_features)
            self.W = np.random.randn(in_features, out_features) * limit
        elif init_method == 'xavier':
            # 适用于 Sigmoid/Tanh: std = sqrt(2/(n_in + n_out))
            limit = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * limit
        else:
            # 普通正态分布
            self.W = np.random.randn(in_features, out_features) * 0.01
            
        self.b = np.zeros((1, out_features))
        
        # 梯度与缓存
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        """ 
        前向传播
        Z = X @ W + b 
        """
        # 仅在允许求导时保存 X 用于反向传播
        if is_grad_enabled():
            self.cache = X
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        反向传播
        :param dZ: 损失函数对当前层输出 Z 的偏导，形状为 (batch_size, out_features)
        :return: 损失函数对输入 X 的偏导 dX，用于传给前一层
        """
        assert self.cache is not None, "Cannot call backward without forward cache. Are you in no_grad mode?"
        X = self.cache
        
        # dL/dW = X.T @ dZ
        self.dW = X.T @ dZ + self.weight_decay * self.W
        
        # dL/db = sum(dZ, axis=0)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        # 传给上一层的梯度: dL/dX = dZ @ W.T
        dX = dZ @ self.W.T
        
        self.cache = None
        
        return dX

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True 

    def forward(self, X):
        # 如果是评估模式或者 p=0，显式将 mask 置为 None
        if not self.training or self.p == 0:
            self.mask = None
            return X
        
        keep_prob = 1 - self.p
        # 生成 mask 并进行缩放 (Inverted Dropout)
        mask = (np.random.rand(*X.shape) > self.p) / keep_prob
        
        # 只有需要梯度时才保存 mask
        if is_grad_enabled():
            self.mask = mask
        else:
            self.mask = None
            
        return X * mask

    def backward(self, dX_out):
        """
        反向传播修正版
        """
        # 如果 self.mask 是 None，说明前向传播没做 drop_out
        if self.mask is None:
            return dX_out
            
        # 否则，按照前向传播的掩码过滤梯度
        return dX_out * self.mask

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
