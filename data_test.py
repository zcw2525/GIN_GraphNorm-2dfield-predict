import pandas as pd
import plotly.express as px

# 加载你的数据 (假设你已经把数据上传到了 Colab)
# df = pd.read_csv('0.1.csv', sep='\s+') # sep='\s+' 处理空格分隔
#
# # 画出流场云图
# fig = px.scatter(df, x='x-coordinate', y='y-coordinate',
#                  color='temperature', # 这里可以换成 mach-number 或 temperature
#                  hover_data=['nodenumber'],
#                  title='CFD Results Visualization')
#
# fig.update_layout(coloraxis_colorbar=dict(title="temperature"))
# fig.show()
import torch
from torch_geometric.utils import is_undirected

# 读取处理后的图
graph = torch.load("dataset/processed/structure_1_old/graph_phi_0.1.pt",
                   weights_only=False)

print("==== Basic Info ====")
print("Number of nodes:", graph.num_nodes)
print("Node feature shape:", graph.x.shape)      # 应该是 (53000, 3)
print("Target shape:", graph.y.shape)            # 应该是 (53000, 4)
print("Edge index shape:", graph.edge_index.shape)

# ----------------------------------------
# 1️⃣ 检查坐标范围
# ----------------------------------------
x_coord = graph.x[:, 0]
y_coord = graph.x[:, 1]
phi_val = graph.x[:, 2]

print("\n==== Coordinate Range ====")
print("x range:", x_coord.min().item(), x_coord.max().item())
print("y range:", y_coord.min().item(), y_coord.max().item())
print("phi unique:", phi_val.unique())

# ----------------------------------------
# 2️⃣ 检查输出范围
# ----------------------------------------
print("\n==== Output Stats ====")
print("Pressure range:", graph.y[:, 0].min().item(), graph.y[:, 0].max().item())
print("Density range:", graph.y[:, 1].min().item(), graph.y[:, 1].max().item())
print("Mach range:", graph.y[:, 2].min().item(), graph.y[:, 2].max().item())
print("Temperature range:", graph.y[:, 3].min().item(), graph.y[:, 3].max().item())

# ----------------------------------------
# 3️⃣ 检查边是否合理
# ----------------------------------------
num_edges = graph.edge_index.shape[1]
print("\n==== Edge Info ====")
print("Total edges:", num_edges)
print("Edges per node (avg):", num_edges / graph.num_nodes)

print("Is undirected:", is_undirected(graph.edge_index))

# ----------------------------------------
# 4️⃣ 检查是否有孤立点
# ----------------------------------------
from torch_geometric.utils import degree
deg = degree(graph.edge_index[0]) #统计每个node作为source_node的出度次数
print("Min degree:", deg.min().item())
print("Max degree:", deg.max().item())
