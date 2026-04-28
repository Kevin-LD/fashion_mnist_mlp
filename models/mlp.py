import numpy as np
import pickle
from core.layers import Linear, Dropout  # 确保 core.layers 中已有 Dropout 类
from core.activations import ReLU, Sigmoid, Tanh

class ThreeLayerMLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, 
                 weight_decay=0.0, activation='relu', dropout_p=0.0, init_method=None):
        """
        三层神经网络模型
        """
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.training = True  # 初始化默认为训练模式
        
        # 1. 自动选择初始化方式
        if init_method is None:
            if activation.lower() == 'relu':
                init_method = 'he'
            else:
                init_method = 'xavier'
        
        # 2. 初始化激活函数
        if activation.lower() == 'relu':
            self.act1 = ReLU()
            self.act2 = ReLU()
        elif activation.lower() == 'sigmoid':
            self.act1 = Sigmoid()
            self.act2 = Sigmoid()
        elif activation.lower() == 'tanh':
            self.act1 = Tanh()
            self.act2 = Tanh()
        else:
            raise ValueError("Unsupported activation function")

        # 3. 初始化 Dropout 层
        self.drop1 = Dropout(p=dropout_p)
        self.drop2 = Dropout(p=dropout_p)

        # 4. 初始化网络层
        self.fc1 = Linear(input_dim, hidden_dim1, weight_decay=weight_decay, init_method=init_method)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, weight_decay=weight_decay, init_method=init_method)
        self.fc3 = Linear(hidden_dim2, num_classes, weight_decay=weight_decay, init_method=init_method)

        # 组织层级结构：FC -> Act -> Dropout
        self.layers = [
            self.fc1, self.act1, self.drop1,
            self.fc2, self.act2, self.drop2,
            self.fc3
        ]

    def forward(self, X):
        """
        前向传播
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dL_dZ_final):
        """
        反向传播
        """
        dout = dL_dZ_final
        # 逆序遍历层并执行 backward
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def train(self):
        """
        切换模型为训练模式：开启 Dropout 并更新属性
        """
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        """
        切换模型为评估模式：关闭 Dropout 并更新属性
        """
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def total_regularization_loss(self):
        """
        计算所有全连接层的 L2 正则化惩罚项
        """
        reg_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear):
                reg_loss += 0.5 * self.weight_decay * np.sum(layer.W**2)
        return reg_loss

    def save_weights(self, path):
        """
        保存权重到二进制文件
        """
        weights = {
            'fc1_W': self.fc1.W, 'fc1_b': self.fc1.b,
            'fc2_W': self.fc2.W, 'fc2_b': self.fc2.b,
            'fc3_W': self.fc3.W, 'fc3_b': self.fc3.b
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        """
        从二进制文件加载权重
        """
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.fc1.W, self.fc1.b = weights['fc1_W'], weights['fc1_b']
        self.fc2.W, self.fc2.b = weights['fc2_W'], weights['fc2_b']
        self.fc3.W, self.fc3.b = weights['fc3_W'], weights['fc3_b']
        print(f"Weights loaded from {path}")

    def get_trainable_layers(self):
        """
        获取所有包含参数的层，供优化器使用
        """
        return [self.fc1, self.fc2, self.fc3]
