"""
----------------------------------------------------------------------
config.py
项目全局配置：路径、时间划分、数据构建参数、模型与训练超参数

约定：
    - 所有路径都基于 PROJECT_ROOT（即本文件所在目录）。
    - 其他模块通过 `from config import XXX` 使用。
    - 想改超参只需改这里，不要去改 src/ 下面的代码。

项目背景（对应论文）：
    李蓉等《基于多模态特征融合的太阳耀斑预报模型》
    中国科学：物理学 力学 天文学, 2024, 54(12): 129611
    模型名 STCNet = Swin Transformer + CNN 双分支多模态融合
----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# 一、项目根目录与各类目录
# ============================================================

# 项目根目录（config.py 所在目录）
PROJECT_ROOT = Path(__file__).parent.resolve()

# ---------- 数据目录：JW-FD（JinWu-Flare Dataset）----------
#
# 数据版本说明（重要）：
#   我们手上的是 Zenodo **v1**（约 10% 代表性子集），论文描述的是 **v2**（完整 15 年）。
#   两者差异：v1 用 split seed=62、只发 Th1000 一档 PNG、无冻结名单 JSON；
#             v2 用 seed=3970、发十档 PNG（200–2000 G，推荐 800 G）、有 split_v2_ar_membership.json。
#   因此本项目的基线与论文报告的 v2 基线、以及原论文的 0.8251 都不可直接比较，
#   必须在本文的设定下自建基线。
DATA_DIR = PROJECT_ROOT / "data"

# JW-FD 原始数据根（含 png_600_Th1000/、flare_labels.csv、split_dataset.py）
JWFD_DIR = DATA_DIR / "raw"

# 600×600 单通道灰度 PNG，307 个活动区子目录，目录名是裸数字（如 1151/）
# 例：png_600_Th1000/1151/AR1151_20110129_000000.png
JWFD_PNG_DIR = JWFD_DIR / "png_600_Th1000"

# ---------- 离线预缩（可选）----------
# 训练/验证时每读一张 600×600 PNG 都要做一次缩放。
#
# 实测 epoch 时间构成（2-epoch 实测反推）：
#     训练阶段 GPU : 约 959 秒（4,967 batch × 193 ms）  ← 占 85–89%，是瓶颈
#     验证 + 开销  : 约 115–164 秒                        ← 占 11–15%
#     合计         : 约 18 分钟/epoch
#
# 用 scripts/prepare_jwfd.py 预缩成 224×224 JPEG 后：
#   - 单张约 7 KB（原 120 KB），读盘量降约 94%，全量约 1.34 GB
#   - 能省的主要是【验证阶段】的数据加载（1–2 分钟/epoch，约 10%）
#   - 【训练阶段基本没提升】（GPU 受限，8 个 worker 已提前备好数据）
#   - 更大的价值是磁盘 I/O 与存储占用
#
# 结论：预缩值得做，但别指望它显著加速；真要提速应上 AMP 混合精度（1.5–2×）。
#
# 用法：
#     python scripts/prepare_jwfd.py          # 生成 data/processed/png_224/
#     然后把下面 USE_PREPROCESSED_PNG 改成 True
JWFD_PNG_224_DIR = DATA_DIR / "processed" / "png_224"

# 是否使用预缩后的 224×224 图片
#   False = 直接用 600×600 PNG（默认）
#   True  = 用 data/processed/png_224/（需先跑 scripts/prepare_jwfd.py）
USE_PREPROCESSED_PNG = False

# 耀斑事件表（1,306 条，列：AR_Number/Begin_Time/Max_Time/End_Time/Flare_Class）
# 主训练流程不直接用它（标签已内置于 CSV），保留供时序分析/多任务实验使用
JWFD_FLARE_LABELS = JWFD_DIR / "flare_labels.csv"

# 官方划分脚本（可复现 8:1:1 AR 级划分，seed=62）
JWFD_SPLIT_SCRIPT = JWFD_DIR / "split_dataset.py"

# 数据索引目录（66 列主数据集 CSV 所在处）
INDEX_DIR = DATA_DIR / "index"
STATS_DIR = DATA_DIR / "stats"                 # 归一化统计量

# 主数据集 CSV：66 列 = 29 特征 + 28 二分类标签 + 7 类别标签 + 2 元数据
# 三份划分由官方 split_dataset.py 生成（AR 级 8:1:1，seed=62），不需要我们重新切分
_CSV_STEM = "solar_flare_dataset_png_Lat60_Lon60_Th1000"
TRAIN_CSV = INDEX_DIR / f"{_CSV_STEM}_train.csv"    # 158,935 行
VAL_CSV = INDEX_DIR / f"{_CSV_STEM}_val.csv"        #  18,262 行
TEST_CSV = INDEX_DIR / f"{_CSV_STEM}_test.csv"      #  23,741 行
FULL_CSV = INDEX_DIR / f"{_CSV_STEM}.csv"           # 200,938 行（全量，仅备用）

# ---------- 旧数据集（SOLAR-STORM2）----------
# 仅供仍引用这些名字的旧脚本使用，新流程**不再使用**。
# 确认 JW-FD 流程跑通后可以删除 data/raw/magnetogram_jpg 与 para_flare.txt 释放空间。
SOLAR_STORM2_DIR = JWFD_DIR
MAGNETOGRAM_DIR = SOLAR_STORM2_DIR / "magnetogram_jpg"
PARA_FLARE_TXT = SOLAR_STORM2_DIR / "para_flare.txt"

# ---------- 预训练权重 ----------
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

# Swin Transformer 主干预训练权重（ImageNet-1K，已改成单通道输入）
# 注意：文件内部结构是 {'model': OrderedDict(...)}，必须取 ["model"] 才能加载！
#       这正是原代码的 bug 所在，详见 src/models/swin.py 的 load_pretrained 说明。
SWIN_PRETRAINED = PRETRAINED_DIR / "swin_tiny_patch4_window7_224.pth"

# 磁场参数 1D-CNN 分支的预训练权重（可选，结构上可直接加载）
CNN_PRETRAINED = PRETRAINED_DIR / "cnn_premodel.pth"

# ---------- 实验产物 ----------
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"

# ============================================================
# 二、数据集与时间划分（严格对应论文 4.1 节）
# ============================================================
#
# 论文原文：
#   "训练集采取的是 2010年5月–2015年10月以及2017年9月–2019年1月的数据。
#    ...训练集在未发生太阳耀斑每天的相同活动区样本中只选择一条记录，
#    使其数量与发生太阳耀斑的样本量接近。经过负样本下采样后，得到训练集
#    正样本数为2988个，负样本数为5073个。测试集采取的是2015年10月–2017年
#    9月的原始数据，其中正样本为196个，负样本为11852个。"
#
# 实测本数据集（73810 个样本）按上述时间段切分：
#   训练段 2010-05-01~2015-10-01 : 正 2792 / 负 54673
#   训练段 2017-09-01~2019-01-31 : 正   49 / 负  4246
#   测试段 2015-10-01~2017-09-01 : 正  147 / 负 11963
# 论文说数据集总计"3184 正 / 70822 负"，但实测标签分布是 2988 正 / 70822 负，
# 负样本完全吻合、正样本差 196（=论文测试集正样本数），故判定论文该处数字有误，
# 以数据实测为准。详见 docs/notes/dataset_build.md。

# ============================================================
# 二、数据集标签配置（JW-FD）
# ============================================================
#
# 【先厘清一个容易误解的点】
# JW-FD 的官方标签定义（论文 2.5 节，图 2）是：
#
#       观测时刻 t 取正类 ⟺ t ∈ [Begin − h, Begin)
#
# 把它解出来就是：Begin ∈ (t, t + h]，即**耀斑将在观测时刻之后的 h 小时内爆发**。
# 也就是说，它同样是"**未来 h 小时窗口**"，和原论文 SOLAR-STORM2 的
# "48 hours starting at the observation" 在**语义上是同一个任务**。
#
# JW-FD 的措辞"严格爆发前窗口"强调的是：观测必须早于爆发（不能用爆发后的帧
# 去"预测"该爆发），这本来就是预测任务的应有之意。
#
# 所以：**任务方向一致，标签口径基本等价**。但两边数字仍不可直接对比，原因是
#      数据集不同（活动区/时段/特征/划分/版本都不同），而不是任务定义不同。
#
# JW-FD 提供 7 个预测窗口 × 4 个 GOES 阈值 = 28 个二分类标签列：
#   窗口 h  ∈ {1, 3, 6, 12, 24, 48, 72} 小时
#   阈值 τ  ∈ {C1.0, M1.0, M5.0, X1.0}
#   列名格式：flare_label_{τ}_{h}hr

# 主任务标签列。默认 ≥M1.0 @ 48h，与原论文"48h 内 ≥M 级"口径对齐。
LABEL_COL = "flare_label_M1.0_48hr"
#
# 实测各候选标签的正样本数（全量 200,938 / 训练 158,935 / 测试 23,741）：
#   flare_label_C1.0_24hr   37,067 / 26,891 / 4,910   ← 正样本最多
#   flare_label_M1.0_48hr   15,134 / 10,834 / 2,087   ← 默认（推荐）
#   flare_label_M1.0_24hr    9,809 /  6,825 / 1,389
#   flare_label_M5.0_48hr    3,475 /  1,995 /   982
#   flare_label_X1.0_24hr      915 /    381 /   360
#   flare_label_X1.0_1hr        45 /     20 /    15   ← 极稀有，仅适合做极端不平衡实验
#
# 改这一行即可切换任务；多任务/多窗口实验后续从这里扩展。

# 标签语义（写进论文时可直接引用）
POSITIVE_MEANS = "观测时刻之后48小时内，同一活动区将发生M级及以上耀斑"

# ---------- 旧的 SOLAR-STORM2 时间划分（已弃用，保留仅为兼容旧脚本）----------
# JW-FD 的三份划分由官方 split_dataset.py 生成（AR 级 8:1:1，seed=62），
# 我们不再自己做时间切分，也不再做负样本下采样。以下变量名保留但已无实际作用。
TRAIN_PERIODS = []          # 已弃用
TEST_PERIODS = []           # 已弃用
VAL_PERIODS = []            # 已弃用
DOWNSAMPLE_NEGATIVE = False  # 已弃用（JW-FD 不做负样本下采样）
DOWNSAMPLE_SEED = 42         # 已弃用
USE_VAL_SPLIT = True         # 已弃用（验证集由官方 val CSV 提供）
VAL_DOWNSAMPLE_NEGATIVE = False  # 已弃用

# ============================================================
# 三、数据相关常量
# ============================================================

# 论文：单通道灰度图，调整尺寸后为 224×224，patch 4×4
IMG_SIZE = 224

# 磁场特征个数。
#   SOLAR-STORM2：10 个 SHARP 参数
#   JW-FD       ：29 个磁场特征（梯度 7 + 中性线 13 + 小波 5 + 磁通量 4）
# 这个数会决定 1D-CNN 分支的输出展平维度（32 * NUM_PARAMS），改动时无需改模型代码。
NUM_PARAMS = 29

# 29 个特征列名（顺序与 CSV 列顺序一致，用于按名取列，避免依赖列号）
# 注意：PNG 分支里 "Gradient min" 恒为 0（8-bit 量化地板），无信息量；
#       如需剔除，把下面这行删掉即可（NUM_PARAMS 会自动跟着变）。
FEATURE_COLS = [
    # 梯度类（7）
    "Gradient mean",
    "Gradient std",
    "Gradient median",
    "Gradient min",          # ⚠️ PNG 分支恒为 0
    "Gradient max",
    "Gradient skewness",
    "Gradient kurtosis",
    # 中性线类（13）
    "NL length",
    "NL no. fragments",
    "NL gradient-weighted length",
    "NL curvature mean",
    "NL curvature std",
    "NL curvature median",
    "NL curvature min",
    "NL curvature max",
    "NL bending energy mean",
    "NL bending energy std",
    "NL bending energy median",
    "NL bending energy min",
    "NL bending energy max",
    # 小波类（5）
    "Wavelet Energy L1",
    "Wavelet Energy L2",
    "Wavelet Energy L3",
    "Wavelet Energy L4",
    "Wavelet Energy L5",
    # 磁通量类（4）
    "Total positive flux",
    "Total negative flux",
    "Total signed flux",
    "Total unsigned flux",
]

# 元数据列名
COL_IMAGE_FILENAME = "image_filename"
COL_IMAGE_PATH = "image_path"

# 一致性自检：特征列数必须与 NUM_PARAMS 相符
assert len(FEATURE_COLS) == NUM_PARAMS, (
    f"FEATURE_COLS 有 {len(FEATURE_COLS)} 列，但 NUM_PARAMS = {NUM_PARAMS}；请保持一致"
)

# 类别数：0 = 无耀斑，1 = 有耀斑
NUM_CLASSES = 2

# ============================================================
# 四、模型结构
# ============================================================

# Swin 主干的型号。
#   'swin_tiny_patch4_window7_224' → embed_dim=96，主干 2756 万参数（原始代码实际用的就是这个）
#   'swin_base_patch4_window7_224' → embed_dim=128，主干 8810 万参数（原 Config 里写的名字，但没生效）
# 依据：model-27/model-29.pth 里 patch_embed.proj.weight 的形状是 (96,1,4,4)，
#       证明学长训练时实际用的是 Tiny。详见 docs/notes/checkpoints.md。
SWIN_VARIANT = "swin_tiny_patch4_window7_224"

# Swin 输出的一维特征维度（= 融合前图像模态的特征长度）
SWIN_FEATURE_DIM = 320

# 磁场参数分支（1D-CNN）输出的特征维度
CNN_FEATURE_DIM = 320

# 分类头 dropout
DROPOUT = 0.3

# 是否使用对比学习（SupConLoss）辅助损失
USE_SUPCON = True

# 对比损失的权重（总损失 = CE + LAMBDA_SUPCON * SupCon）
# 原代码写的是 (CE + SupCon)/2，等价于 LAMBDA_SUPCON=1.0
LAMBDA_SUPCON = 1.0

# SupCon 温度系数
SUPCON_TEMPERATURE = 0.1

# ============================================================
# 五、训练超参数
# ============================================================

BATCH_SIZE = 32

# 训练轮数。
#
# ⚠️ JW-FD 与 SOLAR-STORM2 的 epoch 概念完全不同：
#     SOLAR-STORM2 : 训练集 6,700 条 → 一个 epoch 只有 210 个 batch，约 46 秒
#     JW-FD        : 训练集 158,935 条 → 一个 epoch 有 4,967 个 batch，约 17 分钟
#                                                 （实测 bs=32 与 bs=64 都是 ~17 分钟，
#                                                   因为 Swin 主干是算力受限，加 batch 无收益）
#
# 换算一下"见过的样本数"：
#     旧：6,700 × 50 epoch = 335,000 次样本更新
#     新：158,935 × 12 epoch = 1,907,000 次样本更新  ← 已经是旧的 5.7 倍
#
# 而且历史经验（SOLAR-STORM2 的 5 种子实验）表明最优 epoch 出现在
# **第 1–3 轮**（容量远超数据导致早期就过拟合）。换成数据量 24 倍的数据集后，
# 最优 epoch 会后移，但也不会需要 50 轮。
#
# 所以取 12 轮：
#     12 × 17 分钟 ≈ 3.4 小时（含验证集评估）
#     配合 PATIENCE=5，通常会提前停，实际约 1.5–2 小时
#
# 如果显存/时间充裕，可以调到 20；如果要跑 5 种子实验，建议配合
# AMP 混合精度（见 README 的"后续优化"），可再快 1.5–2 倍。
NUM_EPOCHS = 12

# 分组学习率（原代码就是分三组，保留这个设计）
LR_SWIN = 2e-4          # Swin 主干（预训练过，用小学习率微调）
LR_CNN = 1e-3           # 磁场参数分支（从头训练，用大学习率）
LR_OTHER = 2e-4         # 其余层（分类头、norm 等）

# 学习率衰减：每 STEPLR_STEP 个 epoch 乘 STEPLR_GAMMA
STEPLR_STEP = 4
STEPLR_GAMMA = 0.5

# 数据加载
#
# NUM_WORKERS 是最影响训练速度的一个参数，不要用 0。
#
# 实测（本机 RTX 5070 Laptop + 16 逻辑核，batch_size=32）：
#     num_workers=0 : 单 batch 数据加载 98 ms   ← 在主进程同步做，GPU 干等
#     num_workers=2 : 单 batch 数据加载 39 ms
#     num_workers=4 : 单 batch 数据加载 22 ms
#     num_workers=8 : 单 batch 数据加载  9 ms   ← 11.4 倍加速
# 而 GPU 前向+反向单 batch 只要 195 ms。
# 所以 num_workers=0 时数据加载占掉了约 1/3 的时间，
# 表现为 CPU 打满 90%、GPU 只有 43%（GPU 在等 CPU 喂数据）。
# 开到 8 之后数据加载(9ms) 远小于 GPU 计算(195ms)，GPU 不再等数据。
#
# 取 8 的理由：16 个逻辑核留一半给主进程与系统，避免和 GPU 抢 CPU。
# 如果 CPU 核较少（比如 4 核），改成 2 或 4。
NUM_WORKERS = 8

PIN_MEMORY = True           # 锁页内存，加速 CPU→GPU 拷贝（仅 CUDA 有效）

# 复用 DataLoader 子进程，避免每个 epoch 重建（NUM_WORKERS>0 时生效）
PERSISTENT_WORKERS = True

# 注意：NUM_WORKERS > 0 时，入口脚本必须有
#     if __name__ == "__main__":
# 保护。因为 Windows 用 spawn 方式启动子进程，子进程会重新导入主模块，
# 没有保护就会递归启动进程并报：
#     RuntimeError: An attempt has been made to start a new process before
#     the current process has finished its bootstrapping phase.
# 本项目的 main.py 已经有该保护，可以放心开多进程。

# ---------- 混合精度（AMP）----------
# 这是目前【唯一】能在量级上提速的优化项（数据加载、batch size 都已到瓶颈）。
#
# 实测（本机 RTX 5070 Laptop，bs=32，含梯度裁剪）：
#     FP32（默认）   : 1.00×（基准）
#     AMP + bf16     : 约 1.5×
#     AMP + fp16     : 约 1.5×
# 并且显存从 3.60 GB 降到约 2.55 GB。
#
# 为什么默认 bf16 而不是 fp16：
#     bf16 与 fp32 有相同的指数范围，不需要 GradScaler，也不容易溢出/NaN
#     （Swin 的注意力里有 exp 运算）。fp16 需要 GradScaler，虽已实现但更易踩坑。
#
# ⚠️ 打开前请确认 GPU 支持：
#     支持 bf16 的架构：Ampere（RTX 30 系）及以上、A100/H100 等
#     本机 RTX 5070 Laptop（Blackwell）支持
#     老卡（如 P100/P40/V100 部分型号）不支持 bf16，请改用 amp_dtype="fp16"
USE_AMP = False         # 默认关闭：保持与之前实验完全一致的行为
AMP_DTYPE = "bf16"      # "bf16" 或 "fp16"

# 早停：验证集指标连续 PATIENCE 轮不提升就停。
#
# ⚠️ 这个值必须随"每轮成本"调整：
#     SOLAR-STORM2 时代一个 epoch 只要 46 秒，PATIENCE=10 代价很小；
#     JW-FD 时代一个 epoch 约 17 分钟，PATIENCE=10 意味着白跑 2.8 小时。
# 取 5：既能容忍正常的指标抖动（JW-FD 验证集 18,262 条，比原来的 1,208 条
# 大 15 倍，波动会更小），又不会浪费太多时间。
PATIENCE = 5

# 早停监控的指标：'auc' / 'tss' / 'f1' / 'loss'
#
# ⚠️ 为什么默认改成 'auc'（原来是 'tss'）
# ------------------------------------------------------------------
# 'tss' 依赖固定阈值 0.5，实测这个值**抖动极大**：
#     两次同配置运行的 val_TSS@0.5 相差 0.04~0.06（0.5227 vs 0.5864 / 0.6140 vs 0.5691），
#     这是"阈值切在哪"造成的噪声，不是模型能力差异。
# 用噪声这么大的指标做早停判断，PATIENCE 很容易被误触发 → 提前停掉。
#
# 'auc' 是阈值无关指标，只看"模型把正负样本排序分得开不开"，
# 稳定性远好于 tss（在 SOLAR-STORM2 上实测 AUC 波动 0.0029，TSS 波动 0.0231，差 8 倍）。
#
# 注意：早停用 AUC 选模型，但**最终报告的 TSS 仍然是在验证集上选阈值后算的**
#      （见 `python main.py test --val-threshold`），两者不冲突。
MONITOR_METRIC = "auc"

# 是否用加权采样器平衡正负样本
#   True  = WeightedRandomSampler（把正负拉到 1:1）
#   False = 普通 shuffle（看到真实先验，正样本仅 6.8%）
#
# ⚠️ 2026-10-17 实测结论：**改成 False 明显更好，所以默认值从 True 改成 False**
#
# 对照实验（同一份代码、同样 3 epoch / AMP，仅采样器不同）：
#
#                    加权采样(True)      不加权(False)      变化
#   验证集 AUC        0.8712              0.8966          +0.025
#   测试集 AUC        0.8567              0.8803          +0.024
#   测试集 TSS        0.5893              0.6545          +0.065
#   测试集 FAR        0.3278              0.3010          -0.027
#   验证集最优阈值     0.10                0.010           —
#
# 原因分析：
#   1. 加权采样把正样本过采样约 14 倍 → 每个正样本每个 epoch 被看 14 次
#      → 模型快速记住正样本 → 泛化能力反而下降（AUC 掉 0.025）
#   2. 训练分布(1:1)与推理分布(6.8%)严重不一致 → 概率尺度偏移
#      → 必须用 0.10 这种"奇怪"的阈值，操作点也更差（FAR 0.33 vs 0.30）
#
# 注意：不加权后模型变得非常"保守"——阈值 0.5 时几乎全预测负类
#      （验证集 TPR 仅 0.014），所以**必须配合验证集选阈值**使用，
#      不能再用固定的 0.5。
#
# 想要 loss 层面对不平衡做补偿，请用下面的 CLASS_WEIGHTS 而不是改采样器 ——
# 加权 loss 不会改变数据分布，对概率校准的破坏小得多。
USE_WEIGHTED_SAMPLER = False

# 类别权重（CrossEntropyLoss 的 weight），None 表示不加权
#   JW-FD 主任务（M1.0@48h）训练集 正 10,834 / 负 148,101 ≈ 1:13.7。
#   实测：不加权（None）已经比"加权采样"好，说明数据量足够，不需要人为平衡。
#   如果后续想试"更强地照顾正样本"，建议先试这里而不是打开 USE_WEIGHTED_SAMPLER：
#       CLASS_WEIGHTS = (0.073, 1.0)     # 负类权重小 → 正类被放大
CLASS_WEIGHTS = None

# ============================================================
# 六、随机种子与设备
# ============================================================

RANDOM_SEED = 42

# None 表示自动选择（有 CUDA 用 CUDA，否则 CPU）
DEVICE = None

# ============================================================
# 七、评估
# ============================================================

# 分类阈值：概率 >= 该值判为正类（有耀斑）
EVAL_THRESHOLD = 0.5

# 阈值扫描用的候选值（同时用于"验证集选阈值"和"测试集诊断扫描"）
#
# ⚠️ 2026-10-17 实测教训：原来的范围是 (0.1 ... 0.9)，
#    结果 jwfd_seed1 在验证集上选出的最优阈值 t* = 0.10 —— **正好卡在区间下边界**。
#    这说明真正的最优点可能比 0.1 还低，即模型的输出概率整体偏低。
#
#    为什么偏低：训练用 WeightedRandomSampler 把正负样本拉到 1:1，
#    但验证/测试集仍是原始分布（正样本仅 6.8%）。模型在"平衡世界"里学，
#    到"不平衡世界"里推理时就会系统性地给出偏低的概率。
#
#    所以把区间往下延伸、并在低端加密采样。
THRESHOLD_SWEEP = (
    0.005, 0.010, 0.015, 0.020, 0.030,
    0.050, 0.080, 0.100, 0.150,
    0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
    0.500, 0.600, 0.700, 0.800,
)

# ============================================================
# 八、Config 类：给模型/训练器传参用
# ============================================================
#
# 为什么既有模块级常量、又有 Config 类？
#     模块级常量（上面的 BATCH_SIZE 等）方便 `from config import BATCH_SIZE` 直接用，
#     和你 CME-Detection-Research 项目的写法一致；
#     Config 类则是为了兼容原始代码 src/models/stcnet.py 里 `Mynet(config)` 的接口，
#     模型需要的是一个小对象而不是整个模块。
#     两者数值完全一致，改超参只需改上面，这个类会自动同步。


class Config:
    """
    把模块级配置打包成对象，供模型和训练器使用。

    用法:
        cfg = Config()
        model = Mynet(cfg, pretrained_path=SWIN_PRETRAINED)
    """

    def __init__(self):
        # ---------- 模型结构 ----------
        self.swin_variant = SWIN_VARIANT
        self.swin_feature_dim = SWIN_FEATURE_DIM
        self.cnn_feature_dim = CNN_FEATURE_DIM
        self.num_classes = NUM_CLASSES
        self.dropout = DROPOUT
        self.usesloss = USE_SUPCON              # 兼容原代码字段名
        self.use_supcon = USE_SUPCON
        self.lambda_supcon = LAMBDA_SUPCON
        self.supcon_temperature = SUPCON_TEMPERATURE

        # ---------- 数据 ----------
        self.img_size = IMG_SIZE
        self.num_params = NUM_PARAMS
        self.batch_size = BATCH_SIZE

        # ---------- 训练 ----------
        self.num_epochs = NUM_EPOCHS
        self.lr_swin = LR_SWIN
        self.lr_cnn = LR_CNN
        self.lr_other = LR_OTHER
        # 兼容原代码字段名
        self.transformer_learning_rate = LR_SWIN
        self.textcnn_learning_rate = LR_CNN
        self.other_learning_rate = LR_OTHER
        self.step_lr_step = STEPLR_STEP
        self.step_lr_gamma = STEPLR_GAMMA
        self.num_workers = NUM_WORKERS
        self.pin_memory = PIN_MEMORY
        self.persistent_workers = PERSISTENT_WORKERS
        self.use_amp = USE_AMP                 # 混合精度开关（trainer 里读）
        self.amp_dtype = AMP_DTYPE             # "bf16" / "fp16"
        self.patience = PATIENCE
        self.monitor_metric = MONITOR_METRIC
        self.use_weighted_sampler = USE_WEIGHTED_SAMPLER
        self.class_weights = CLASS_WEIGHTS
        self.seed = RANDOM_SEED
        self.device = DEVICE

        # ---------- 评估 ----------
        self.eval_threshold = EVAL_THRESHOLD
        self.threshold_sweep = THRESHOLD_SWEEP

    def __repr__(self):
        return (
            f"Config(swin={self.swin_variant}, epochs={self.num_epochs}, "
            f"bs={self.batch_size}, lr=({self.lr_swin},{self.lr_cnn},{self.lr_other}), "
            f"supcon={self.use_supcon}, monitor={self.monitor_metric})"
        )


# ============================================================
# 九、便捷自检
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("项目路径检查")
    print("=" * 70)
    checks = {
        "项目根目录": PROJECT_ROOT,
        "磁图目录": MAGNETOGRAM_DIR,
        "参数表": PARA_FLARE_TXT,
        "Swin 预训练权重": SWIN_PRETRAINED,
        "CNN 预训练权重": CNN_PRETRAINED,
    }
    for name, p in checks.items():
        flag = "OK " if Path(p).exists() else "缺失"
        print(f"  [{flag}] {name:16s} {p}")

    print()
    print("时间划分：")
    print(f"  训练段: {TRAIN_PERIODS}")
    print(f"  验证段: {VAL_PERIODS if USE_VAL_SPLIT else '（不设验证集）'}")
    print(f"  测试段: {TEST_PERIODS}")
    print()
    print(f"模型: {SWIN_VARIANT}")
    print(f"batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, 监控指标={MONITOR_METRIC}")
