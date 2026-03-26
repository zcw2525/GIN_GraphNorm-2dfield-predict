import os
import torch
from torch_geometric.data import Dataset
from utils import load_norm_stats
class CFDDataset(Dataset):
    def __init__(self, root_dir, phi_list, normalization=True):
        """

        :param root_dir: processed dataset directory
        :param phi_list: list of phi values to load
        :param normalization: whether nomalize tatget
        """
        super().__init__()
        self.root_dir = root_dir
        self.phi_list = phi_list
        self.normalization = normalization

        #加载归一化参数
        self.mean, self.std = load_norm_stats()

    def len(self):
        return len(self.phi_list)

    def get(self, idx):
        phi =  self.phi_list[idx]
        path = os.path.join(self.root_dir, f'graph_phi_{phi}.pt')
        data = torch.load(path, weights_only=False)

        if self.normalization:
            data.y = (data.y - self.mean) / self.std

        return data

