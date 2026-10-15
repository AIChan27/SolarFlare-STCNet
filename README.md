# SolarFlare-STCNet

太阳耀斑预报项目 —— 复现论文《基于多模态特征融合的太阳耀斑预报模型》的 STCNet 模型。

> 李蓉, 黄鑫, 等. 基于多模态特征融合的太阳耀斑预报模型[J].
> 中国科学: 物理学 力学 天文学, 2024, 54(12): 129611.

STCNet = **Swin Transformer**（处理太阳光球纵向磁图）+ **1D-CNN**（处理 10 个 SHARP 磁场特征参数）
两个模态融合，预报未来 **48 小时**内是否发生 **M 级及以上**太阳耀斑。


## 一、这个项目是什么

学长给的原始代码（`SwinT-cnn/`）能看出论文的模型结构，但存在几个问题：

1. **跑不起来**：磁场参数分支的输入形状是 4 维，`Conv1d` 直接报错；
2. **预训练权重静默失效**：352 MB 的权重一个参数都没加载上（还不报错）；
3. **缺少数据构建环节**：官方数据集只有"磁图 + 参数表"，
   论文里"按时间划分 + 负样本下采样"那一步的代码完全不存在。

本项目在保留原模型结构的前提下：

- 修掉了全部 bug（详见 [docs/troubleshooting/bugs.md](docs/troubleshooting/bugs.md)）；
- 补上了缺失的数据构建环节（详见 [docs/notes/dataset_build.md](docs/notes/dataset_build.md)）；
- 按你现有项目 `CME-Detection-Research/` 的结构重新组织了目录。

**现在可以端到端跑通了**，实测指标见第七节。


## 二、目录结构

```
SolarFlare-STCNet/
│
├── main.py                        # 项目主入口，按阶段调用 pipeline
├── config.py                      # 全局配置（路径、时间划分、超参数）
├── setup_project.py               # 目录初始化脚本，clone 后运行一次
├── README.md                      # 项目说明（本文件）
├── .gitignore                     # Git 忽略规则
│
├── src/                           # 源代码
│   ├── __init__.py
│   │
│   ├── data/                      # 数据 pipeline（按阶段拆分为多个文件）
│   │   ├── __init__.py
│   │   ├── preprocess.py          # 阶段一：从原始数据构建训练/验证/测试索引 CSV
│   │   └── dataset.py             # 阶段二：SolarFlareDataset + DataLoader
│   │
│   ├── models/                    # 模型定义
│   │   ├── __init__.py
│   │   ├── swin.py                # Swin Transformer 主干（单通道输入）
│   │   └── stcnet.py              # STCNet 完整模型 + 1D-CNN 分支 + SupConLoss
│   │
│   ├── train/                     # 训练与评估
│   │   ├── __init__.py
│   │   ├── trainer.py             # 训练循环、分组学习率、早停
│   │   └── evaluate.py            # 测试集评估、混淆矩阵、指标计算、阈值扫描
│   │
│   └── utils/                     # 通用工具
│       ├── __init__.py
│       ├── paths.py               # 路径管理（统一获取各目录）
│       └── visualization.py       # 可视化（训练曲线、ROC、混淆矩阵、预测样例）
│
├── data/                          # 数据目录（整体加入 .gitignore）
│   └── index/                     # 三份索引 CSV（阶段一产物）
│       ├── trainflare1.csv
│       ├── validflare1.csv
│       └── testflare1.csv
│
├── pretrained/                    # 预训练权重
│   ├── swin_tiny_patch4_window7_224.pth    # 109 MB，Swin 主干
│   └── cnn_premodel.pth                    # 0.4 MB，1D-CNN 分支（默认不用）
│
├── docs/                          # 项目文档
│   ├── notes/
│   │   ├── dataset_build.md       # 数据集构建笔记（官方说明 + 实测统计 + 划分方案）
│   │   ├── checkpoints.md         # 每个 .pth 文件装了什么（详细解析）
│   │   └── architecture.md        # 模型结构与数据流
│   └── troubleshooting/
│       └── bugs.md                # 原代码 6 个 bug 的定位、复现与修复
│
├── notebooks/                     # Jupyter 探索（待补充）
│
└── experiments/                   # 实验产物（整体加入 .gitignore）
    ├── logs/                      # 训练历史 CSV
    ├── checkpoints/               # 模型权重
    ├── figures/                   # 论文配图、ROC、混淆矩阵、预测样例
    └── reports/                   # 指标 JSON
```


## 三、快速开始

### 0. 环境准备

```bash
conda activate cme          # 或任意装好 torch 的环境
pip install torch torchvision scikit-learn pandas pillow matplotlib
```

本机实测环境：Python 3.12.14 / PyTorch 2.14.1+cu132 / torchvision 0.29.1 / RTX 5070 Laptop。

### 1. 初始化目录

```bash
python setup_project.py
```

### 2. 检查环境与数据

```bash
python main.py check
```

会打印依赖版本、数据文件是否存在、磁图数量（应为 73810）、
以及模型能否成功构建（含预训练权重加载统计）。

### 3. 构建数据索引

```bash
python main.py build-data
```

产出三份 CSV 到 `data/index/`，耗时约 10 秒。

### 4. 冒烟测试（先确认流程通）

```bash
python main.py train --smoke
```

训练集截断为 200 条、只跑 1 个 epoch，约 12 秒。

### 5. 正式训练

```bash
python main.py train --tag stcnet
```

全量 6700 条 × 50 epoch，本机单卡（RTX 5070 Laptop）实测每 epoch 约 85 秒，
50 epoch 约 70 分钟；有早停（默认连续 10 轮 TSS 不提升就停），实际通常更短。

### 6. 测试集评估

```bash
python main.py test --tag stcnet --sweep
```

输出 ACC / F1 / TPR / TNR / FAR / TSS / AUC + 混淆矩阵，并画图。
`--sweep` 会额外做阈值扫描。


## 四、常用命令

| 命令 | 作用 |
|---|---|
| `python main.py check` | 检查环境、数据、模型 |
| `python main.py build-data` | 构建索引 CSV |
| `python main.py build-data --no-downsample` | 构建但不下采样负样本 |
| `python main.py train` | 正式训练 |
| `python main.py train --smoke` | 冒烟测试（小样本 1 epoch） |
| `python main.py train --epochs 10 --tag exp1` | 指定 epoch 数和实验标签 |
| `python main.py train --no-supcon` | 关闭对比学习（消融实验） |
| `python main.py train --no-pretrain` | 不加载预训练权重（消融实验） |
| `python main.py test --tag exp1` | 评估指定实验 |
| `python main.py test --ckpt <路径> --sweep` | 评估指定权重 + 阈值扫描 |
| `python main.py all` | 一条龙：build-data → train → test |


## 五、数据

数据集：**SOLAR-STORM2**（空间环境预警和AI技术跨学科创新工作组提供）

- 下载：<https://tianchi.aliyun.com/dataset/74780>
- 内容：73,810 张太阳光球纵向磁图 + 10 个 SHARP 磁场参数 + 耀斑标签
- 时间范围：2010-05-01 ~ 2019-01-26
- 标签：1 = 48h 内发生 M 级及以上耀斑；0 = 没有

**注意：官方数据集没有划分训练/测试集**，论文里的划分是作者自己做的。
本项目按论文的方案实现，详见 [docs/notes/dataset_build.md](docs/notes/dataset_build.md)。

数据默认放在工作区的 `../SwinT-cnn/data/`（7.4 GB，为免搬动没有移动）。
如果想换个位置，改 `config.py` 里的 `SOLAR_STORM2_DIR`。


## 六、时间划分（严格对应论文 4.1 节）

```
训练集：2010-05-01 ~ 2015-10-01  以及  2017-09-01 ~ 2019-01-31
验证集：2015-01-01 ~ 2015-10-01    ← 论文没提验证集，本项目自行设计
测试集：2015-10-01 ~ 2017-09-01   （原始数据，不下采样）
```

负样本下采样（论文规则："每天、相同活动区只选一条记录"）：

| | 正样本 | 负样本 | 合计 |
|---|---|---|---|
| 训练集 | 2,358 | 4,342 | **6,700** |
| 验证集 | 483 | 725 | **1,208** |
| 测试集 | 147 | 11,963 | **12,110** |

**为什么验证集取 2015 年而不是训练段 B 的尾部**：
实测训练段 B（2017-09 ~ 2019-01）的 49 个正样本全部集中在 2017-09 一个月，
之后一个正样本都没有（太阳活动极小期），切出的验证集会退化成 0 正 / 187 负，
TSS 恒为 0，早停完全失效。改用训练段 A 的尾部后，
验证集正负比 1:1.5（与论文训练集接近），训练集仅损失 17% 的正样本。


## 七、实测结果

用 **3 个 epoch** 的演示训练（`--tag demo`，约 4 分钟）在测试集上得到：

| 指标 | 本项目 (3 epoch) | 论文 STCNet | 说明 |
|---|---|---|---|
| **TSS** | **0.8432** | 0.8251 | 论文最看重的指标，本项目更高 |
| TPR(报准率) | **0.9592** | 0.9227 | 147 个真耀斑中命中 141 个 |
| TNR | 0.8840 | 0.9024 | 略低于论文 |
| FAR(虚报率) | 0.1160 | 0.0976 | 略高于论文 |
| ACC | 0.8849 | 0.9027 | |
| F1 | 0.1683 | 0.1463 | |
| AUC | 0.9711 | — | |

混淆矩阵：TN=10575, FP=1388, FN=6, TP=141

阈值扫描显示 **阈值 0.6 时 TSS 最优（0.8507）**，此时 TPR=0.9320、FAR=0.0813，
即在保持高报准率的同时把虚报率进一步压低。

> **注意**：两边测试集不完全相同（本项目实测正样本 147 个，论文写 196 个），
> 所以这个对比只能作为量级参考，不能当作严格复现。
> 而且只训了 3 个 epoch，正式训练 50 epoch 后指标应该还有提升空间。


## 八、每个 .pth 权重文件装了什么

**完整解析见 [docs/notes/checkpoints.md](docs/notes/checkpoints.md)**，这里给结论速查：

| 文件 | 大小 | 内容 | 本项目用途 |
|---|---|---|---|
| `swin_tiny_patch4_window7_224.pth` | 109 MB | `{'model': OrderedDict(190 项)}`，`patch_embed=(96,1,4,4)`，即 Swin-Tiny 单通道预训练 | ✅ **使用**，初始化主干 |
| `swin_base_patch4_window7_224.pth` | 336 MB | 同上但是 Base，`patch_embed=(128,1,4,4)` | ❌ 不用（学长实际训的是 Tiny） |
| `cnn_premodel.pth` | 0.4 MB | `OrderedDict(16 项)`，1D-CNN 分支权重，但形状是 **Conv2d** 的 `(16,1,1,1)` | ❌ 默认不用（形状不匹配 Conv1d） |
| `model-27.pth` / `model-29.pth` | 各 105.7 MB | **完整模型对象**（`torch.save(model)`），Mynet 类，27,636,956 参数 | 📌 保留作参照，证明实际用 Tiny |

三个关键发现：

1. **预训练权重外层包了一层 `{'model': ...}`**，必须取 `["model"]` 才能加载
   —— 这正是原代码最大的 bug，漏了这一步导致 352 MB 权重静默失效。
2. **`patch_embed=(96,1,4,4)` 证明学长实际用的是 Swin-Tiny 而非 Base**，
   本项目据此设置 `SWIN_VARIANT`，`main.py check` 报出的参数量
   27,636,956 与训练产物**完全一致**。
3. **`model-27/29.pth` 是完整模型对象**，新版 PyTorch 的 `weights_only=True`
   默认值会导致加载失败，本项目改存 `state_dict` 并保留对旧格式的兼容。


## 九、原代码修了什么

**详细复现与修复过程见 [docs/troubleshooting/bugs.md](docs/troubleshooting/bugs.md)**：

| # | 问题 | 严重程度 | 修复位置 |
|---|---|---|---|
| 1 | 预训练权重静默加载失败（0 个参数加载上） | **致命** | `stcnet.py::load_pretrained` |
| 2 | 磁场参数输入 4 维，`Conv1d` 报错 | **致命** | `dataset.py`、`stcnet.py::CNN` |
| 3 | 主干型号三方不一致（配置/代码/权重） | 高 | `config.SWIN_VARIANT` |
| 4 | `torch.load` 新版 `weights_only` 报错 | 中 | `trainer.py`、`main.py` |
| 5 | `forward` 里每 batch 重复加载 336 MB 权重 | 中 | `stcnet.py::__init__` |
| 6 | matplotlib 中文标题缺字 | 低 | `visualization.py` |


## 十、引用

```bibtex
@article{李蓉2024基于多模态,
  title   = {基于多模态特征融合的太阳耀斑预报模型},
  author  = {李蓉 and 黄鑫 and others},
  journal = {中国科学: 物理学 力学 天文学},
  volume  = {54},
  number  = {12},
  pages   = {129611},
  year    = {2024}
}
```

数据集引用：

> the dataset of Active-Region magnetogram and flare provided by
> Space Environment Warning and AI Technology Interdisciplinary Innovation
> Working Group [Xin H, et al., 2018]

- Xin H, Huaning W, Long X, et al. Deep Learning Based Solar Flare Forecasting
  Model. I. Results for Line-of-sight Magnetograms[J]. Astrophysical Journal,
  2018, 856(1): 7.
