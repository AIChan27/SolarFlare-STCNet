# ============================================================
# 磁场参数分支：1D-CNN
# ============================================================

# from __future__ import annotations
#   作用：让类型注解（如 list[dict]）在 Python 3.9 及以下也能用
from __future__ import annotations

import torch.nn as nn

class CNN(nn.Module):
    """
    处理 N 个磁场特征参数的一维卷积网络（论文 3.1 节第二部分）。

    ⚠️ 特性数 N 是**可配置的**：
        - SOLAR-STORM2（原论文数据集）：N = 10 个 SHARP 参数
        - JW-FD（现在用的数据集）      ：N = 29 个磁场特征
                                      （梯度 7 + 中性线 13 + 小波 5 + 磁通量 4）
        N 由 config.NUM_PARAMS 决定，通过 Mynet 传进来。
        **不能写死成 10**，否则换数据集后 flatten 维度会与 fc 不匹配。

    结构（对应论文 3.3 节："卷积层使用大小为 3 的卷积核, 分别具有 16 个和
    32 个输出通道, 后跟批归一化(BatchNorm)加速网络的收敛并提高模型的性能,
    以及用于增强模型非线性能力的 ReLU 激活函数"）：
        Conv1d(1  → 16, kernel=3, stride=1, padding=1) + BN + ReLU
        Conv1d(16 → 32, kernel=3, stride=1, padding=1) + BN + ReLU
        Flatten  → Linear(32*N → 320)

    为什么 kernel=3 要配 padding=1
    ------------------------------
    一维卷积的输出长度公式：
        L_out = ⌊(L_in + 2*padding − kernel_size) / stride⌋ + 1
    代入 kernel_size=3、stride=1：
        padding=0 → L_out = L_in − 2   长度变了，flatten 变成 32*(N-2)，fc 必须跟着改
        padding=1 → L_out = L_in       长度不变，flatten 仍是 32*N，与 fc(32*N, 320) 匹配
    论文只说了 kernel=3，没提 padding。为了让"N 个特征进 → N 个位置出"
    这个语义成立（并保持 320 维输出不变），padding 取 1。

    为什么用 kernel=3 而不是 1×1
    ---------------------------
    kernel=3 使每个输出位置能看到相邻 3 个特征，显式建模特征之间的局部组合
    关系（JW-FD 的 29 个特征是按物理类别连续排列的：梯度、中性线、小波、
    磁通量，所以相邻位置确有物理关联）；
    1×1 卷积只能在每个特征位置上各自独立变换，看不到邻域。

    输入输出形状（重要）
    ------------------
        输入  x : (B, 1, N)   ← 必须是 3 维！Conv1d 只接受 2D 或 3D 输入
        输出    : (B, 320)

    Parameters
    ----------
    num_params : int
        输入特征个数 N。默认 10（兼容原论文的 SOLAR-STORM2 设定）。
        JW-FD 应传 29。
    out_dim : int
        输出特征维度，默认 320（与 Swin 分支的特征维度对齐）。
    """

    def __init__(self, num_params: int = 10, out_dim: int = 320):
        super(CNN, self).__init__()
        if num_params <= 0:
            raise ValueError(f"num_params 必须为正整数，收到 {num_params}")

        self.num_params = num_params
        self.out_dim = out_dim

        # 通道数 1 → 16，卷积核大小 3（论文 3.3 节），padding=1 保持序列长度 N
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        # 通道数 16 → 32，同样 kernel=3, padding=1，长度仍为 N
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # 展平后是 32 通道 × 长度 N = 32*N 维，映射到 out_dim 维特征
        # N=10 → 320（原论文）；N=29 → 928（JW-FD）
        self.fc = nn.Linear(32 * num_params, out_dim)

    def forward(self, x):
        # x 的期望形状：(B, 1, num_params)
        # 原代码在这里写了 x.squeeze(2)，那是想压掉一个多余维度，
        # 但原 dataset 产出的是 (B,1,10,1)，squeeze(2) 压的是"长度 10"那一维
        # （squeeze 只在维度大小为 1 时才生效），所以压不掉，直接崩：
        #   RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d,
        #                 but got input of size: [B, 1, 10, 1]
        # 现在 src/data/dataset.py 保证输入就是 (B,1,N)，不需要 squeeze。
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)      # (B, 32, N) → (B, 32*N)
        x = self.fc(x)                 # (B, out_dim)
        return x

