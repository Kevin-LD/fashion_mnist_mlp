import numpy as np

class _LRScheduler:
    def __init__(self, optimizer, last_step=-1):
        """
        学习率调度器基类
        :param last_step: 上一次调度的索引，初始为 -1 确保第一次 step() 后变为 0
        """
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.last_step = last_step
        # 第一次执行以同步初始 lr
        self.step()

    def step(self):
        """
        统一的接口：推进进度并更新 optimizer 的 lr
        """
        self.last_step += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        """
        由子类实现具体的计算逻辑
        """
        raise NotImplementedError

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        """
        阶梯衰减: lr = initial_lr * (gamma ^ (step // step_size))
        :param step_size: 每隔多少个 step 衰减一次
        :param gamma: 衰减倍率
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        # 使用下取整计算当前处于第几个阶梯
        return self.initial_lr * (self.gamma ** (self.last_step // self.step_size))

class LinearLR(_LRScheduler):
    def __init__(self, optimizer, total_steps):
        """
        线性衰减: 从 initial_lr 线性降至 0
        """
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        factor = 1.0 - (self.last_step / self.total_steps)
        return self.initial_lr * max(0.0, factor)

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_steps):
        """
        余弦退火: lr = 0.5 * initial_lr * (1 + cos(pi * step / total_steps))
        """
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        factor = 0.5 * (1 + np.cos(np.pi * self.last_step / self.total_steps))
        return self.initial_lr * factor
