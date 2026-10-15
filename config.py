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

# ---------- 数据目录 ----------
DATA_DIR = PROJECT_ROOT / "data"

# 数据集存放位置。
# 注意：SOLAR-STORM2 原始数据有 7.4 GB，为免搬动，默认指向工作区里已有的位置。
#       如果你把 data/ 搬进本项目，改成 DATA_DIR / "raw" 即可。
SOLAR_STORM2_DIR = DATA_DIR / "raw"

# 磁图目录（73,810 张 .jpg 灰度图，平铺在一个目录里）
MAGNETOGRAM_DIR = SOLAR_STORM2_DIR / "magnetogram_jpg"

# 磁场特征参数 + 耀斑标签表（73,810 行 × 12 列）
PARA_FLARE_TXT = SOLAR_STORM2_DIR / "para_flare.txt"

# 数据构建产物的输出目录
INDEX_DIR = DATA_DIR / "index"                 # 三个 split 的索引 CSV
RAW_JPG_DIR = DATA_DIR / "raw_jpg"             # 按类别分目录的图片（软链接或复制）
STATS_DIR = DATA_DIR / "stats"                 # 归一化统计量

# 数据集索引文件（三列结构：图片路径 + 10 个参数 + 标签，见 src/data/dataset.py 说明）
TRAIN_CSV = INDEX_DIR / "trainflare1.csv"
VAL_CSV = INDEX_DIR / "validflare1.csv"
TEST_CSV = INDEX_DIR / "testflare1.csv"

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

# 训练集的时间段（列表，元素为 (起, 止)，闭区间，格式 YYYYMMDD）
TRAIN_PERIODS = [
    ("20100501", "20151001"),   # 第 1 段
    ("20170901", "20190131"),   # 第 2 段
]

# 测试集的时间段（论文：2015年10月–2017年9月的原始数据）
TEST_PERIODS = [
    ("20151001", "20170901"),
]

# 负样本下采样：每天的、每个活动区的负样本只保留 1 条（论文原文规则）
#   True  = 按论文规则下采样（训练集约剩 5000 多条负样本）
#   False = 不下采样，保留全部负样本（正负比约 1:19.6）
DOWNSAMPLE_NEGATIVE = True

# 负样本下采样的随机种子（保证可复现）
DOWNSAMPLE_SEED = 42

# ---------- 验证集 ----------
# 论文没有提验证集，但训练需要它来做早停/选最优权重。
# 做法：从训练集里再按时间切出一段做验证，避免时间泄漏。
#   USE_VAL_SPLIT = True  → 从训练集中划出 VAL_PERIODS 作为验证集
#   USE_VAL_SPLIT = False → 不设验证集（用测试集评估，训练时用训练集 loss 早停）
USE_VAL_SPLIT = True

# 验证集时间段。
#
# 【重要】为什么验证段取自 2015 年，而不是训练段 B 的尾部：
#   实测训练段 B（2017-09 ~ 2019-01）的全部 49 个正样本都集中在 2017-09 这一个月，
#   2017-10 之后到 2019-01 一个正样本都没有（太阳活动极小期）。
#   所以论文的划分本身无法从训练段 B 中切出含两类样本的验证集
#   —— 之前试过用 2018-01~2019-01 作验证段，结果验证集是 0 正 / 187 负，
#   单类别验证集算不出 TSS/F1，早停会完全失效。
#
#   因此改为取训练段 A 的尾部（2015 年全年）：
#       验证集   : 正 483 / 负 723（下采样后），约 1:1.5，与论文训练集比例接近
#       训练集   : 正样本从 2841 降到 2358（只损失约 17%）
#   这样既保证了时间上不泄漏（训练 2010-05~2014-12，验证 2015-01~2015-09），
#   又保证了验证集两类样本都够多，TSS 才稳定。
VAL_PERIODS = [
    ("20150101", "20151001"),
]

# 验证集的负样本是否也下采样（同样按"每天每活动区一条"）
VAL_DOWNSAMPLE_NEGATIVE = True

# ============================================================
# 三、数据相关常量
# ============================================================

# 论文：单通道灰度图，调整尺寸后为 224×224，patch 4×4
IMG_SIZE = 224

# 磁场特征参数个数（para_flare.txt 的第 2–11 列）
NUM_PARAMS = 10

# 类别数：0 = 无耀斑，1 = 有耀斑
NUM_CLASSES = 2

# 标签语义（对应官方数据集说明与论文 3.1 节）
#   1 = 从磁图观测时刻起的 48 小时内，至少发生 1 个 M 级或更大耀斑
#   0 = 48 小时内没有发生 M 级或更大耀斑
POSITIVE_MEANS = "48小时内至少发生1个M级及以上耀斑"

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
NUM_EPOCHS = 50

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

# 早停：验证集指标连续 PATIENCE 轮不提升就停
PATIENCE = 10

# 早停监控的指标：'f1' / 'tss' / 'loss'
MONITOR_METRIC = "tss"      # 太阳耀斑预报里 TSS 是最有参考价值的指标

# 是否用加权采样器平衡正负样本
#   True  = WeightedRandomSampler（推荐；论文做了负样本下采样，但残余不平衡仍在）
#   False = 普通 shuffle
USE_WEIGHTED_SAMPLER = True

# 类别权重（CrossEntropyLoss 的 weight），None 表示不加权
#   训练集下采样后约 正:负 = 2988:5073，约 1:1.7，可以不加权
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

# 阈值扫描用的候选值
THRESHOLD_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

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
