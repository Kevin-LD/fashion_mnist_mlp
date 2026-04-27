import os
import sys
import shutil
import tempfile
import argparse
import unittest
from unittest.mock import patch

# 动态将项目根目录添加到系统路径中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# train 函数和数据集准备函数
from train import train
from datasets.dataset import prepare_data

class TestTrainOverfit(unittest.TestCase):
    def setUp(self):
        """
        测试前的准备工作：创建一个临时目录用于存放模型权重和日志。
        """
        self.test_dir = tempfile.mkdtemp()
        
        # 指向真实数据集的路径
        self.real_data_path = os.path.join(PROJECT_ROOT, 'data')
        self.num_samples = 16  # 我们只切出 16 个真实的图像样本用于过拟合测试

    def tearDown(self):
        """
        测试结束后的清理工作：删除存放日志和模型的临时目录。
        """
        shutil.rmtree(self.test_dir)

    @patch('train.prepare_data')
    def test_overfit_real_small_dataset(self, mock_prepare_data):
        """
        核心测试：读取数据集，切出 16 个样本，验证模型能否对这 16 个真实图片 100% 拟合。
        """
        # 1. 获取数据
        try:
            (X_all, y_all), _, _ = prepare_data(self.real_data_path, val_ratio=0.0)
        except FileNotFoundError:
            self.fail(f"测试失败：找不到真实数据集，请确认 {self.real_data_path} 下存在 ubyte 文件！")
            
        # 2. 从真实数据中切出一小部分
        X_small = X_all[:self.num_samples]
        y_small = y_all[:self.num_samples]
        
        # 3. 拦截 train.py 里的 prepare_data，将我们切片好的真实数据喂给它
        # 将验证集设为与训练集相同的数据，这样 Val Acc 100% 就意味着完美记住了这些数据
        mock_prepare_data.return_value = (
            (X_small, y_small),  # (X_train, y_train)
            (X_small, y_small),  # (X_val, y_val)
            None                 # test_set
        )

        # 4. 构造模拟的命令行参数
        args = argparse.Namespace(
            data_path=self.real_data_path,  # 这里传入什么都可以，因为已经被 mock 拦截了
            save_dir=self.test_dir,
            resume='',
            hidden1=128,
            hidden2=64,
            activation='relu',
            epochs=80,                      # 给予足够的轮数确保 Loss 能降到底
            batch_size=self.num_samples,    # Full-batch 训练 (Batch Size = 16)
            val_batch_size=self.num_samples,
            lr=0.05,                        # 使用适中的学习率
            momentum=0.9,
            weight_decay=0.0,               # 关闭正则化 (L2)
            scheduler='constant',
            step_size=10,
            gamma=0.1,
            no_save_history=False
        )

        # 5. 启动训练
        best_val_acc, history = train(args)

        # -------------------------------------------
        # 测试断言 (Assertions)
        # -------------------------------------------
        
        # A. 检查最终准确率是否达到了 95% (由于同源数据，Val Acc 代表记忆率)
        self.assertGreaterEqual(
            best_val_acc, 0.95, 
            "模型未能在小批量数据上达到 95% 以上准确率"
        )

        # B. 检查最终的 Data Loss 是否降得足够低 (接近 0)
        final_loss = history['epoch']['train_data_loss'][-1]
        self.assertLess(
            final_loss, 0.1, 
            f"最后的训练数据 Loss ({final_loss}) 偏高"
        )
        
        # C. 检查模型权重和历史记录是否正常输出
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'best_model.pkl')), "权重未保存。")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'history.json')), "日志未保存。")
        
        print("\n[✓] 模型成功过拟合了 16 张真实的图像样本")

if __name__ == '__main__':
    # 在命令行中运行时输出详细的测试结果
    unittest.main(verbosity=2)
