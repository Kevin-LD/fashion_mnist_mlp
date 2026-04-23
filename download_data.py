import os
import gzip
import urllib.request
import sys

def progress_bar(count, block_size, total_size):
    """
    进度条回调函数
    count: 已下载的数据块数量
    block_size: 数据块的大小
    total_size: 文件的总大小
    """
    if total_size <= 0:
        return
    
    # 计算进度
    progress = count * block_size
    percent = min(progress * 100 / total_size, 100.0)
    
    # 构造进度条字符串
    bar_length = 30
    filled_length = int(bar_length * percent / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # 使用 \r 实现原地刷新
    sys.stdout.write(f'\r进度: |{bar}| {percent:.1f}%')
    sys.stdout.flush()

def download_fashion_mnist(target_dir='./data'):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in files:
        file_path = os.path.join(target_dir, file)
        
        # 1. 下载
        if not os.path.exists(file_path):
            print(f"正在下载 {file}...")
            # 传入 reporthook 参数
            urllib.request.urlretrieve(base_url + file, file_path, reporthook=progress_bar)
            print() # 下载完成后换行
        else:
            print(f"{file} 已存在，跳过下载。")

        # 2. 解压
        unzipped_path = file_path.replace('.gz', '')
        if not os.path.exists(unzipped_path):
            print(f"正在解压 {file}...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(unzipped_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            print(f"{unzipped_path} 已存在，跳过解压。")

    print("\n所有数据已准备就绪！")

if __name__ == "__main__":
    download_fashion_mnist()
    