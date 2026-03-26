import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from model import AppendEvs
from config import SAVE_DIR, DATA_DIR,PHI_LIST, TRAIN_PHI,KNN_K,num_evs

os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# 1. 读取 .dat 文件
# ===============================
def load_dat_file(filepath):
    """
    读取 .dat 文件
    默认按空格分隔
    返回:
        pos  : (N,2) tensor
        y    : (N,4) tensor  [p, rho, Ma, T]
    """

    df = pd.read_csv(filepath, sep=r"\s+", engine="python")

    # 如果第一列是 node number，删除它
    if "nodenumber" in df.columns[0].lower():
        df = df.iloc[:, 1:]

    # 假设列顺序:
    # x, y, pressure, density, mach-number, temperature
    x = df.iloc[:, 0].values
    y_coord = df.iloc[:, 1].values

    pressure = df.iloc[:, 2].values
    density = df.iloc[:, 3].values
    mach = df.iloc[:, 4].values
    temperature = df.iloc[:, 5].values

    pos = torch.tensor(np.stack([x, y_coord], axis=1), dtype=torch.float)
    target = torch.tensor(
        np.stack([pressure, density, mach, temperature], axis=1),
        dtype=torch.float
    )

    return pos, target


# ===============================
# 2. 构造 edge_index（只做一次）
# ===============================
def build_edge_index(pos):
    edge_index = knn_graph(pos, k=KNN_K, loop=False)
    edge_index = to_undirected(edge_index)
    return edge_index


# ===============================
# 3. 主预处理流程
# ===============================
def preprocess():
    print("Start preprocessing...")

    transform = AppendEvs(num_evs)
    graphs = []#存所有工况的下的图
    all_train_targets = []

    edge_index = None
    pos_reference = None

    #构造图Data对象，并存在graphs列表里
    for phi in PHI_LIST:
        filename = f"{phi}.dat"
        filepath = os.path.join(DATA_DIR, filename)

        pos, target = load_dat_file(filepath)

        # 第一次构造 edge（只用运行一次）,以及构建特征向量
        if edge_index is None:
            pos_reference = pos
            edge_index = build_edge_index(pos)
            torch.save(edge_index, os.path.join(SAVE_DIR, "edge_index.pt"))

            # ⭐ 只算一次 EV,x随便设计的，因为不用x
            temp_data = Data(x=torch.zeros((pos.shape[0],3)),
                             edge_index=edge_index)
            temp_data = transform(temp_data)
            torch.save(temp_data.gin_EVs, os.path.join(SAVE_DIR, "gin_EVs.pt"))


        # 构造输入特征: [x, y, phi]
        phi_tensor = torch.full((pos.shape[0], 1), phi)
        x_input = torch.cat([pos, phi_tensor], dim=1)

        gin_EVs = torch.load(os.path.join(SAVE_DIR, "gin_EVs.pt"))

        data = Data(x=x_input, edge_index=edge_index, y=target)

        #直接赋值，不用计算
        data.gin_EVs = gin_EVs
        graphs.append(data)

    # ===============================
    # 4. 计算标准化参数（只用 TRAIN_PHI）
    # ===============================
    train_graphs = []
    #只选择train的工况下的图来参与统计量计算
    for g, phi in zip(graphs, PHI_LIST):
        if phi in TRAIN_PHI:
            train_graphs.append(g)

    all_train_targets = torch.cat([g.y for g in train_graphs], dim=0)

    mean = all_train_targets.mean(dim=0)
    std  = all_train_targets.std(dim=0)

    torch.save({"target_mean": mean, "target_std": std},
               os.path.join(SAVE_DIR, "norm_stats.pt"))

    # 保存每个 graph,未做归一化处理
    for i, phi in enumerate(PHI_LIST):
        torch.save(graphs[i],
                   os.path.join(SAVE_DIR, f"graph_phi_{phi}.pt"))

    print("Preprocessing finished.")
    print("Mean:", mean)
    print("Std:", std)


if __name__ == "__main__":
    preprocess()