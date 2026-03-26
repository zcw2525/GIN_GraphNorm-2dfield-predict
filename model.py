# 1. 内置库
from copy import deepcopy
import numpy as np
# 2. 第三方库
import torch as pt
import torch.nn as nn
from torch.nn import Dropout, Linear, Parameter, ReLU, ModuleList, Sequential
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.inits import zeros, ones, normal
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, scatter
import pytorch_lightning as ptl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
# 3. 自定义库
from utils import load_norm_stats, inverse_z_score
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigs


class AppendEvs(BaseTransform):
    def __init__(self, num_evs):
        #减1是因为后续会再加一个rest向量，来留出位置
        self.num_evs = num_evs - 1

    def __call__(self, data):
        # 1. edge_index（shape=[2, E]）→ 稀疏邻接矩阵（shape=[N,N]）
        topology = to_scipy_sparse_matrix(data.edge_index)
        # 2. 稀疏→稠密（shape仍为[N,N]，但变成全量存储的二维数组）方便做后续的concate
        topology = topology.todense()

        # 3. 拼接列/行（shape变为[N+1, N+1]）
        topology = np.concatenate((topology, 1e-5 * np.ones( (topology.shape[0], 1)) ), axis=1)
        topology = np.concatenate((topology, 1e-5 * np.ones( (1, topology.shape[1])) ), axis=0)

        # 4. 稠密→稀疏（shape仍为[N+1,N+1]，转回高效存储）
        topology = csr_matrix(topology, dtype=np.float32)

        gin_conv = deepcopy(topology)

        # 对卷积运算因子特征值与特征向量的分解
        # which = 'LM'返回的是降序排列的
        # eigs的要满足 k < N-1，也就是最大N-2
        evalues, evectors = eigs(gin_conv, k=min(self.num_evs, gin_conv.shape[0]-2), which="LM", tol=1e-8, maxiter=100000)

        # 将特征向量按特征值的大小的索引，进行按列的从小到大的排行
        # evectors.shape[N+1,k]u
        evectors = evectors[:, np.argsort(evalues.real)].real
        evectors = pt.tensor(evectors, dtype=pt.float32)

        # 通过补全行[:-1]的符号固定特征向量的符号方向
        evectors = evectors[:-1] * pt.sign(evectors[-1])

        # 归一化的ones元素[1/sqrt(N), .....]
        ones = pt.ones(evectors.shape[0]) / np.sqrt(evectors.shape[0])

        #------源代码逻辑-------
        # # evectors.transpose(0,1) @ ones是求的特征的加权系数，shape[k,], 再broadcast*evector
        # evectors = ((evectors.transpose(0, 1) @ ones) + 1e-4) * evectors
        # rest = ones - pt.sum(evectors, dim=1)

        # 2. 标准正交投影：ones在evectors列空间上的投影（核心简化！）
        # 公式：Proj = V @ (V^T @ o) （V=evectors，o=ones）
        proj = evectors @ ((evectors.transpose(0, 1) @ ones) + 1e-4)  # shape [N,]
        # 3. 计算rest向量（ones - 标准投影）
        rest = ones - proj  # shape [N,]

        # 最后按列拼接三个张量，生成维度统一的谱域特征
        # 先是做了特殊处理的特征向量，然后再是补齐维度的0向量，最后把rest换为列向量补上
        data.gin_EVs = pt.cat((
            evectors,
            pt.zeros(evectors.shape[0], (self.num_evs - evectors.shape[1])),
            rest.reshape(-1,1)),
            dim=1
        )
        return data

class GraphNormv2(nn.Module):
    def __init__(self, num_evs, hidden_dim, eps: float = 1e-5):
        super().__init__()
        self.num_evs = num_evs
        self.hidden_dim = hidden_dim #节点的隐藏层维度
        self.eps = eps #防止方差在分母时为0

        #定义可学习参数(nn.Parameter表示参数会被优化器优化)
        self.weight = nn.Parameter(pt.empty(self.hidden_dim))
        self.bias = nn.Parameter(pt.empty(self.hidden_dim))
        # 设计一个shape[num_evs, hidden_dim]维度的可学习参数
        # 后面将可学习参数融合进contribution里
        self.EV_scales = nn.Parameter(pt.empty(self.num_evs, self.hidden_dim))

        self.reset_parameters()#初始化所有参数

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        normal(self.EV_scales, 0, 0.1) #先初步设为一个小范围的数，model慢慢调节

    def forward(self, x, evectors, batch):
        """
        GraphNormV2前向传播：结合谱域特征向量的拓扑信息，计算每个节点专属的归一化均值/标准差
        Args:
            x: 批次所有节点的特征张量，shape=[N, hidden_dim]
            evectors: 批次所有节点的谱域特征向量（拓扑信息），shape=[N, num_evs]
            batch: 节点所属图的批次索引，shape=[N,]
        Returns:
            归一化后的节点特征，shape=[N, hidden_dim]
        """
        # 1. 统计批次内的图数量（标量，int类型）
        batch_size = int(batch.max()) + 1  # shape=()

        # 2. 计算谱域特征向量对节点特征的基础贡献值
        # evectors.transpose(0,1): [num_evs, N] @ x: [N, hidden_dim] → [num_evs, hidden_dim]
        contribute = evectors.transpose(0, 1) @ x  # shape=[num_evs, hidden_dim]

        # 3. 为基础贡献值加入可学习缩放参数（按"谱向量-特征维度"配对调节）
        # self.EV_scales: [num_evs, hidden_dim]，与contribute → [num_evs, hidden_dim]
        scaled_contribute = (1 + self.EV_scales) * contribute  # shape=[num_evs, hidden_dim]

        # 4. 计算每个节点的专属谱域均值（融合拓扑+可学习权重）
        # evectors: [N, num_evs] @ scaled_contribute: [num_evs, hidden_dim] → [N, hidden_dim]
        # 区别普通GraphNorm：此处mean是每个节点-特征维度的专属值，而非同图共享均值
        mean = evectors @ scaled_contribute  # shape=[N, hidden_dim]

        # 5. 节点特征去中心化（减去专属均值）
        out = x - mean  # shape=[N, hidden_dim]

        # 6. 按图分组计算去中心化特征的方差（图级统计量）
        # out.pow(2): [N, hidden_dim] → scatter按batch分组求均值 → [batch_size, hidden_dim]
        var = scatter(out.pow(2), batch, dim=0, dim_size=batch_size, reduce='mean')  # shape=[batch_size, hidden_dim]

        # 7. 计算图级标准差并映射到节点级（保证维度匹配）
        # (var + eps).sqrt(): [batch_size, hidden_dim] → index_select按batch映射 → [N, hidden_dim]
        std = (var + self.eps).sqrt().index_select(0, batch)  # shape=[N, hidden_dim]

        # 8. 最终归一化：可学习缩放+偏置
        # self.weight/self.bias: [hidden_dim]，广播后与out/std逐元素运算 → [N, hidden_dim]
        return self.weight * out / std + self.bias  # shape=[N, hidden_dim]

#定义单独的残差连接模块
class Res_con(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 设置可训练的alpha参数
        self.alpha = Parameter(pt.tensor(0.3, dtype=pt.float32))
        self.W_1 = Parameter(
            nn.init.normal_(pt.empty(self.hidden_dim, self.hidden_dim),0, 0.04))
        self.W_2 = Parameter(
            nn.init.normal_(pt.empty(self.hidden_dim, self.hidden_dim),0, 0.04))

    def forward(self, x, x0):
        # x是当前层的embedding，y则是原始数据经过encoder的数据y = encoder(x)
        # 后续要改为层间连接
        return self.alpha * x @ self.W_1 + (1 - self.alpha) * x0 @ self.W_2



class GIN_Module(ptl.LightningModule):
    def __init__(self, config):
        super().__init__()
        #配置hyperparameters
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_evs = config['num_evs']
        self.lr = config['lr'] #lr = learning_rate
        self.norm_type = config['normalization']
        self.use_res =  config['residual']
        self.dropout = Dropout(p=config['dropout'])
        self.relu = ReLU()
        self.EV_key = 'gin_EVs'

        self.encoder = Linear(self.input_dim, self.hidden_dim)
        self.decoder = Linear(self.hidden_dim, self.output_dim)

        # 加载归一化参数，用于test_step做反归一化
        mean, std = load_norm_stats()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        #设置可学习的参数，给graphnormv2使用
        self.graphnorm_lambda = Parameter(pt.tensor(0.5, dtype=pt.float32))
        self.norm_gamma = Parameter(pt.tensor(1.0, dtype=pt.float32))
        self.norm_beta = Parameter(pt.tensor(0.0, dtype=pt.float32))

        # 配置不同的归一化,其中x是当前层的embedding，data是原始数据
        if self.norm_type == 'graphnorm':
            self.extract_data = lambda x, data : (x,)
            self.normalization = ModuleList([GraphNorm(self.hidden_dim) for _ in range(self.num_layers)])
        elif self.norm_type == 'graphnormv2':
            self.extract_data = lambda x, data : (x, data[self.EV_key], data.batch)
            self.normalization = ModuleList([GraphNormv2(self.num_evs, self.hidden_dim) for _ in range(self.num_layers)])
        else:
            #什么都不做的情况下，就直接返回原数据
            self.extract_data = lambda x, data : (x,)
            self.normalization = None

        #配置是否有残差连接
        if self.use_res is not None:
            self.res_con =  ModuleList([
                Res_con(self.hidden_dim) for _ in range(self.num_layers)
            ])

        #定义图卷积层
        self.convs = ModuleList([GINConv(
            Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
            )
        ) for _ in range(self.num_layers)])

        self.save_hyperparameters(config, logger=False)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #encoder将节点embedding升维,x0作为原始数据传入残差连接层
        x = self.encoder(x)
        x0 = x

        for i in range(self.num_layers):
            # 1.GINConv
            x = self.convs[i](x, edge_index)
            # 2.normalization
            if self.normalization is not None:
                x = self.normalization[i](*self.extract_data(x, data))
            # 3.res_con
            if self.use_res:
                x = self.res_con[i](x, x0)
            #4.relu+dropout
            x = self.dropout(self.relu(x))

        return self.decoder(x)

    def training_step(self, batch, batch_index):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)
        #return 的 loss 即为要优化的loss
        return loss

    def test_step(self, batch, batch_index):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)

        # 反归一化，便于后续求各项指标
        pred_real = inverse_z_score(pred, self.mean, self.std)
        y_real = inverse_z_score(batch.y, self.mean, self.std)

        feature_names = ["pressure", "density", "mach-number", "temperature"]
        # per-feature metrics ( for 4 features)
        for i, name in enumerate(feature_names):

            feature_squared_error_i = (pred_real[:, i] - y_real[:, i]) ** 2

            num = pt.sum(feature_squared_error_i)
            den = pt.sum(y_real[:, i] ** 2)

            #各种metrics，RelL2， RMSE, PSNR
            rel_l2_i = pt.sqrt(num / den)
            mse_i = pt.mean(feature_squared_error_i)
            rmse_i = pt.sqrt(mse_i)
            max_i = y_real[:, i].max()
            psnr_i = 10 * pt.log10((max_i ** 2) / mse_i)

            self.log(f"rmse_{name}", rmse_i, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)
            self.log(f"psnr_{name}", psnr_i, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)
            self.log(f"rel_l2_{name}", rel_l2_i, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)




    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }













