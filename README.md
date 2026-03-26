# GIN_GraphNorm-2dfield-predict

**基于 2D CFD 的流场预测模型**  
主要采用 **Graph Isomorphism Network (GIN) + GraphNorm** 实现对二维计算流体力学（CFD）流场的预测。

本项目将二维流场数据转化为图结构，利用 GIN 结合 GraphNorm 进行高效的节点特征更新，实现对速度场、压力场等物理量的端到端预测。项目基于 PyTorch Lightning 构建，结构清晰、训练高效，适合科研复现与二次开发。

---

## ✨ 主要特性

- **核心模型**：GIN + GraphNorm（有效缓解图神经网络的过平滑问题）
- **任务类型**：2D CFD 流场预测（速度场 / 压力场 / 涡量场等）
- **数据处理**：完整的预处理 → 图数据集 → DataLoader 流水线
- **训练框架**：PyTorch Lightning（支持分布式、自动日志、Checkpoint）
- **包含脚本**：数据预处理、训练、测试、配置管理全覆盖
- **日志完整**：支持 TensorBoard / Lightning Logs 可视化

---

## 📁 项目结构

```bash
GIN_GraphNorm-2dfield-predict/
├── config.py                 # 超参数配置
├── preprocess.py             # CFD 数据转图结构预处理
├── dataset_dataloader.py     # 数据集与 DataLoader 定义
├── model.py                  # GIN + GraphNorm 模型实现
├── train.py                  # 训练主脚本
├── test.py                   # 测试 / 推理脚本
├── data_test.py              # 测试数据加载工具
├── utils.py                  # 通用工具函数
├── lightning_logs/           # Lightning 训练日志
├── logs/                     # 自定义运行日志
├── running_log.txt           # 运行记录
├── .gitignore
└── README.md
