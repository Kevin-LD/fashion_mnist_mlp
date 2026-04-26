# 全局变量，控制是否需要保存梯度所需的缓存
_grad_enabled = True

def is_grad_enabled():
    """获取当前是否允许梯度计算"""
    return _grad_enabled

def set_grad_enabled(mode: bool):
    """设置当前梯度计算状态"""
    global _grad_enabled
    _grad_enabled = mode

class no_grad:
    """
    上下文管理器，用于在推理时禁用梯度计算缓存，节省内存。
    使用方法:
        with no_grad():
            output = model.forward(X)
    """
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)
