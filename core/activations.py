import numpy as np

class ReLU:
    def __init__(self):
        # 缓存 Z 用于反向传播
        self.cache = None

    def forward(self, Z):
        """
        ReLU(z) = max(0, z)
        """
        self.cache = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        """
        dL/dZ = dA * (1 if Z > 0 else 0)
        """
        Z = self.cache
        dZ = dA * (Z > 0).astype(float)
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
        self.cache = A
        return A

    def backward(self, dA):
        """
        dL/dZ = dA * f(z) * (1 - f(z)) = dA * A * (1 - A)
        """
        A = self.cache
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
        self.cache = A
        return A

    def backward(self, dA):
        """
        dL/dZ = dA * (1 - tanh(z)^2) = dA * (1 - A^2)
        """
        A = self.cache
        return dA * (1 - A**2)
