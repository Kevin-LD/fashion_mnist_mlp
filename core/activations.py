import numpy as np
from core.grad_mode import is_grad_enabled

class ReLU:
    def __init__(self):
        # 缓存 Z 用于反向传播
        self.cache = None

    def forward(self, Z):
        """
        ReLU(z) = max(0, z)
        """
        if is_grad_enabled():
            self.cache = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        """
        dL/dZ = dA * (1 if Z > 0 else 0)
        """
        assert self.cache is not None, "Cannot call backward without forward cache. Are you in no_grad mode?"
        Z = self.cache
        dZ = dA * (Z > 0).astype(float)
        # 反向传播后清空缓存，释放内存（良好习惯）
        self.cache = None 
        return dZ

class Sigmoid:
    def __init__(self):
        # 缓存输出 A，Sigmoid 的导数可以用 A(1-A) 表示
        self.cache = None

    def forward(self, Z):
        """
        f(z) = 1 / (1 + exp(-z))
        """
        A = 1 / (1 + np.exp(-Z))
        if is_grad_enabled():
            self.cache = A
        return A

    def backward(self, dA):
        """
        dL/dZ = dA * f(z) * (1 - f(z)) = dA * A * (1 - A)
        """
        assert self.cache is not None, "Cannot call backward without forward cache."
        A = self.cache
        self.cache = None
        return dA * A * (1 - A)

class Tanh:
    def __init__(self):
        # 缓存输出 A，Tanh 的导数可以用 1 - A^2 表示
        self.cache = None

    def forward(self, Z):
        """
        f(z) = tanh(z)
        """
        A = np.tanh(Z)
        if is_grad_enabled():
            self.cache = A
        return A

    def backward(self, dA):
        """
        dL/dZ = dA * (1 - tanh(z)^2) = dA * (1 - A^2)
        """
        assert self.cache is not None, "Cannot call backward without forward cache."
        A = self.cache
        self.cache = None
        return dA * (1 - A**2)
