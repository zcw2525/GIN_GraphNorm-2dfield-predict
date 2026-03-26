import os
import torch
from config import SAVE_DIR
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device:{device}")
    return device

def cal_time(runtime):
    minutes = int(runtime // 60)
    seconds = runtime % 60

    return minutes, seconds

def load_norm_stats(root_dir=SAVE_DIR):
    """
    加载在preprocess.py文件中处理并存入的归一化参数
    """
    stats_path = os.path.join(root_dir, 'norm_stats.pt')
    stats = torch.load(stats_path)

    mean = stats['target_mean']
    std = stats['target_std']
    return mean, std

def inverse_z_score(x, mean, std):
    """
    反归一化 (inverse z-score)

    x_norm = (x - mean) / std
    x = x_norm * std + mean
    """
    return x * std + mean
