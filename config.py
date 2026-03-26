# ===============================
# Configuration
# ===============================
DATA_DIR = "./dataset/raw/structure_1"              # 原始 .dat 文件目录
SAVE_DIR = "dataset/processed/structure_1"  # 处理后保存目录

 # 所有工况
PHI_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]

# 训练 / 测试划分
TRAIN_PHI = [0.1, 0.3, 0.4]
TEST_PHI  = [0.2, 0.5]

KNN_K = 6                            # KNN邻居数
num_layers = 4