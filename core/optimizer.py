import numpy as np

class SGD:
    def __init__(self, layers, lr=0.1, momentum=0.9):
        """
        随机梯度下降优化器 (PyTorch 风格)
        :param layers: 模型中包含参数的层列表
        :param lr: 初始学习率
        :param momentum: 动量系数 (0.0 为标准 SGD)
        """
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        
        # 初始化动量速度
        self.velocities = []
        for layer in self.layers:
            self.velocities.append({
                'dW': np.zeros_like(layer.W),
                'db': np.zeros_like(layer.b)
            })

    def step(self):
        """
        执行参数更新: v = mu * v + g; theta = theta - lr * v
        """
        for i, layer in enumerate(self.layers):
            if self.momentum > 0.0:
                # 速度更新
                self.velocities[i]['dW'] = self.momentum * self.velocities[i]['dW'] + layer.dW
                self.velocities[i]['db'] = self.momentum * self.velocities[i]['db'] + layer.db
                # 权重更新
                layer.W -= self.lr * self.velocities[i]['dW']
                layer.b -= self.lr * self.velocities[i]['db']
            else:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
