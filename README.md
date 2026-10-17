# SolarFlare-STCNet

太阳耀斑预报项目 —— 复现论文《基于多模态特征融合的太阳耀斑预报模型》的 STCNet 模型，
并在新的 **JW-FD 数据集**上重建基线。

> 李蓉, 黄鑫, 等. 基于多模态特征融合的太阳耀斑预报模型[J].
> 中国科学: 物理学 力学 天文学, 2024, 54(12): 129611.

STCNet = **Swin Transformer**（处理太阳光球纵向磁图）+ **1D-CNN**（处理磁场特征参数）
两个模态融合，预报未来 **48 小时**内是否发生 **M 级及以上**太阳耀斑。


## 一、这个项目是什么

学长给的原始代码（`SwinT-cnn/`）能看出论文的模型结构，但存在几个问题：

1. **跑不起来**：磁场参数分支的输入形状是 4 维，`Conv1d` 直接报错；
2. **预训练权重静默失效**：352 MB 的权重一个参数都没加载上（还不报错）；
3. **缺少数据构建环节**：原数据集只有"磁图 + 参数表"，论文里划分那一步的代码完全不存在。

本项目在保留原模型结构的前提下：

- **修掉了全部 bug**（详见 [docs/troubleshooting/bugs.md](docs/troubleshooting/bugs.md)）；
- **迁移到 JW-FD 数据集**（307 活动区 / 200,938 样本 / 29 个磁场特征 / 官方活动区级划分），
  原来 SOLAR-STORM2 时代的数据构建代码已标记作废；
- 按 `CME-Detection-Research/` 的结构重新组织了目录；
- **修正了三处实验方法学问题**（阈值泄漏、早停判据、加权采样器），见 9.3 节。

**现在可以端到端跑通**，JW-FD 基线结果见第七节。

> **换机器先跑**：`python scripts\check_env.py` —— 检查依赖包 / GPU / 数据是否就绪。
> 在 Tesla P40 等 Pascal 卡上还有额外注意事项，见 [docs/notes/run_on_p40.md](docs/notes/run_on_p40.md)。


## 二、目录结构

```
SolarFlare-STCNet/
│
├── main.py                        # 主入口：check / check-data / train / test / all
├── config.py                      # 全局配置（路径 / 特征列 / 标签列 / 超参数）
├── setup_project.py               # 目录初始化 + 关键文件检查
├── README.md                      # 项目说明（本文件）
├── .gitignore                     # Git 忽略规则
│
├── src/
│   ├── data/
│   │   ├── dataset.py             # SolarFlareDataset（66 列按列名读取）+ DataLoader
│   │   └── preprocess.py          # 【已作废】SOLAR-STORM2 时代的数据构建脚本，仅存档
│   ├── models/
│   │   ├── swin.py                # Swin-Tiny 主干（单通道 in_chans=1）
│   │   ├── cnn.py                 # 1D-CNN 磁场特征分支（29 特征）
│   │   ├── stcnet.py              # STCNet 完整模型 + 预训练权重加载
│   │   └── supconloss.py          # 监督对比损失 SupConLoss
│   ├── train/
│   │   ├── trainer.py             # 训练循环、分组学习率、AMP、早停
│   │   └── evaluate.py            # 指标、混淆矩阵、阈值扫描、验证集选阈值
│   └── utils/
│       ├── paths.py               # 路径管理
│       └── visualization.py       # 训练曲线、ROC、混淆矩阵、预测样例
│
├── scripts/                       # 独立脚本（不参与训练主流程）
│   ├── check_env.py               # ★ 换机器第一步：检查依赖包 / GPU / 数据
│   ├── prepare_jwfd.py            # 离线预缩 600×600 → 224×224（可选，省 I/O）
│   ├── run_multiseed.py           # 一键多种子：train → test → 汇总
│   ├── aggregate_seeds.py         # 多种子 mean ± std 汇总表
│   ├── diag_threshold.py          # 阈值曲线诊断（验证集 vs 测试集对照）
│   └── inspect_jwfd.py            # 数据集探索/统计工具
│
├── data/                          # 数据目录（**整体 .gitignore**，体积 23 GB）
│   ├── index/                     # 官方提供的索引 CSV（66 列）
│   │   ├── solar_flare_dataset_png_Lat60_Lon60_Th1000.csv          # 全量
│   │   ├── ..._train.csv          # 158,935 行
│   │   ├── ..._val.csv            #  18,262 行
│   │   └── ..._test.csv           #  23,741 行
│   ├── raw/
│   │   ├── png_600_Th1000/{AR号}/AR*_*.png    # 307 个活动区 / 200,938 张
│   │   ├── flare_labels.csv                   # 1,306 条耀斑事件表
│   │   ├── split_dataset.py                   # 官方划分脚本
│   │   └── para_flare.txt                     # 【遗留】SOLAR-STORM2 参数表，已不使用
│   └── processed/png_224/         # 预缩产物（可选，默认 USE_PREPROCESSED_PNG=False）
│
├── pretrained/                    # 预训练权重
│   ├── swin_tiny_patch4_window7_224.pth    # 109 MB，Swin 主干（使用）
│   └── cnn_premodel.pth                    # 0.4 MB，1D-CNN 分支（形状不匹配，未用）
│
├── docs/
│   ├── notes/
│   │   ├── run_on_p40.md          # ★ 在 Tesla P40 上运行的注意事项
│   │   ├── architecture.md        # 模型结构与数据流
│   │   ├── checkpoints.md         # 每个 .pth 文件装了什么
│   │   └── dataset_build.md       # SOLAR-STORM2 时代的数据集笔记（存档）
│   ├── reports/
│   │   ├── JW-FD迁移与代码改动总账.md   # ★ 所有代码差异的完整记录
│   │   └── 实验总结与后续计划.md         # 实验总结、Q&A、后续规划
│   └── troubleshooting/
│       └── bugs.md                # 原代码 6 个 bug 的定位、复现与修复
│
├── notebooks/                     # Jupyter 探索（待补充）
│
└── experiments/                   # 实验产物（部分入库，见下方说明）
    ├── logs/                      # 训练历史 CSV + 多种子日志
    ├── checkpoints/               # 模型权重
    ├── figures/                   # 训练曲线、ROC、混淆矩阵、预测样例
    └── reports/                   # 指标 JSON + 多种子汇总 md
```

**哪些目录进 git、哪些不进**（见 `.gitignore`）：

| 目录 | 是否入库 | 原因 |
|---|---|---|
| `data/` | ❌ | 23 GB 数据集，绝对不提交 |
| `pretrained/` | ❌ | 109 MB 权重，单独拷贝 |
| `experiments/checkpoints/` | ❌ | 每个 106 MB，仓库会爆 |
| `experiments/logs/` | ❌ | 训练过程文件 |
| `experiments/reports/` | ❌ | 指标 JSON（但 `seed_summary.md` 这类汇总会手动保留） |
| **`experiments/figures/`** | ✅ **入库** | 单张 30–460 KB，方便直接看结果图（当前约 4 MB） |
| `src/`、`scripts/`、`docs/`、`*.py` | ✅ | 代码与文档 |

> 换机器时**需要手动拷贝**的只有两样：`data/`（或至少 `data/index/` + `data/raw/png_600_Th1000/`）
> 和 `pretrained/swin_tiny_patch4_window7_224.pth`。


## 三、快速开始

### 0. 环境准备（**换机器务必先做这一步**）

```powershell
conda activate 你的环境名
python scripts\check_env.py          # 只检测，不安装任何东西
```

`check_env.py` 会报告：

| 检查项 | 内容 |
|---|---|
| **Python / conda** | 解释器绝对路径、conda 环境名，**并检测"激活的环境"和"实际执行的 python"是否同一个** |
| **依赖包** | torch / torchvision / numpy / pandas / scikit-learn / matplotlib / Pillow 是否齐、版本是否够 |
| **GPU** | CUDA 是否可用、显卡型号、计算能力，并按架构给出"能否用 `--amp`"的建议 |
| **数据** | 三份索引 CSV、PNG 目录、Swin 预训练权重是否到位（次要） |

退出码：`0` = 依赖与 GPU 都就绪；`1` = 有缺失（会打印对应的 pip 安装命令）。

缺包时手动安装：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib pillow
```

> **CUDA 版本怎么选**：用 `nvidia-smi` 看右上角 `CUDA Version`。
> - **Tesla P40 / 其他 Pascal 卡**：必须用 **cu121 或 cu124**（CUDA 13 已不支持 Pascal）
> - RTX 30/40/50 系：可以用更新的 cu126/cu128
>
> 详见 [docs/notes/run_on_p40.md](docs/notes/run_on_p40.md)。

本机开发环境（仅作参考，**不是要求**）：
Python 3.12.14 / PyTorch 2.14.1+cu132 / torchvision 0.29.1 / RTX 5070 Laptop 12 GB。

### 1. 初始化目录

```powershell
python setup_project.py
```

### 2. 检查环境与模型

```powershell
python main.py check
```

会打印依赖版本、JW-FD 数据目录是否存在、活动区/PNG 数量（应为 307 / 200,938）、
三份 CSV 行数，以及模型能否成功构建（含预训练权重加载统计）。

### 3. 校验数据集

```powershell
python main.py check-data
```

校验内容：行数、66 列完整性、28 个标签列分布、图片路径存在性（抽样）、
**以及活动区级划分是否真的没有交叉**（防止同一活动区跨训练/测试的泄漏）。

> 说明：JW-FD 官方已经用 `split_dataset.py` 按活动区做了 8:1:1 划分（seed=62）
> 并直接给出 train/val/test 三份 CSV，so **不需要再自己切分**——
> 原来 SOLAR-STORM2 时代的 `main.py build-data` 已改为 `check-data`。

### 4. 冒烟测试（先确认流程通）

```bash
python main.py train --smoke
```

训练集截断为 200 条（正负各 100）、只跑 1 个 epoch。
注意：即使冒烟测试也要过一遍 18,262 条的验证集，约 2 分钟。

### 5.（可选）离线预缩

```bash
python scripts/prepare_jwfd.py
```

把 600×600 PNG 预缩成 224×224 JPEG（**实测 7 KB/张，全量约 1.34 GB**；
4 进程约 128 张/秒 → 全量约 26 分钟，8 进程更快），
然后改 `config.py` 里 `USE_PREPROCESSED_PNG = True`。

**收益要如实看**（实测拆解）：一个 epoch 约 18 分钟，其中**训练阶段 GPU 占 85–89%**，
验证只占 11–15%。所以预缩：

- 能省的主要是**验证阶段的数据加载**（1–2 分钟/epoch，约 10%）
- **训练阶段基本没有提升**（GPU 是瓶颈，8 个 worker 已把数据提前备好）
- 更大的价值是**磁盘 I/O 和存储占用**（120 KB → 7 KB，降约 94%）

> 想真正提速，应该走 **AMP 混合精度**（1.5–2×），而不是预缩。

### 6. 正式训练

```bash
python main.py train --tag jwfd
```

- 训练集 **158,935** 条，batch 32 → 训练阶段约 **16 分钟**（GPU 前向+反向）
- 加上验证集评估（18,262 条，约 2–3 分钟），约 **18 分钟/epoch**
- 默认 `NUM_EPOCHS=12`、`PATIENCE=5`：最快约 6 轮早停 ≈ 1.8 小时，
  最慢跑满 12 轮 ≈ 3.6 小时

> **⚠️ 与 SOLAR-STORM2 时代的成本对比**：那时训练集 6,700 条、46 秒/epoch。
> 现在数据量是 24 倍，**不要沿用 50 epoch 的旧习惯**——配置里已改成 12。

### 7. 测试集评估

```bash
python main.py test --tag jwfd --sweep
```

输出 ACC / F1 / TPR / TNR / FAR / TSS / AUC + 混淆矩阵，并画图。
`--sweep` 会额外做阈值扫描。


## 四、常用命令

### 4.1 环境与自检

| 命令 | 作用 |
|---|---|
| `python scripts\check_env.py` | ★ 检查依赖包 / GPU / 数据是否就绪（换机器第一件事） |
| `python main.py check` | 检查环境、数据目录、模型构建 |
| `python main.py check-data` | 校验数据集：行数 / 列 / 标签 / 图片 / 活动区零重叠 |

### 4.2 训练

| 命令 | 作用 |
|---|---|
| `python main.py train --tag exp1` | 正式训练 |
| `python main.py train --smoke` | 冒烟测试（200 条样本 1 epoch） |
| `python main.py train --epochs 3 --tag exp1` | 指定轮数（JW-FD 上 3 轮已足够，见第七节） |
| `python main.py train --amp --tag exp1` | 混合精度 bf16（**仅 Ampere 及以上；P40 不能用**） |
| `python main.py train --seed 1 --tag base` | 指定种子（自动拼成 `base_seed1`） |
| `python main.py train --no-supcon` | 关闭对比学习（消融） |
| `python main.py train --no-pretrain` | 不加载预训练权重（消融） |
| `python main.py train --no-weighted-sampler` | 关闭加权采样器（**当前默认已是关闭**） |

### 4.3 评估

| 命令 | 作用 |
|---|---|
| `python main.py test --tag exp1` | 评估（固定阈值 0.5，兼容旧行为） |
| **`python main.py test --tag exp1 --val-threshold`** | ★ **推荐**：验证集选阈值 → 测试集只评估一次（避免泄漏） |
| `python main.py test --tag exp1 --val-threshold --threshold-metric f1` | 换选阈值的判据（默认 tss） |
| `python main.py test --ckpt <路径> --sweep` | 评估指定权重 + 测试集阈值扫描（**仅诊断，不能用于报告**） |
| `python scripts\diag_threshold.py --tag exp1` | 画出"验证集 vs 测试集"的 TSS-阈值曲线，判断是尖峰还是平台 |

### 4.4 多种子实验

| 命令 | 作用 |
|---|---|
| `python scripts\run_multiseed.py --tag jwfd --seeds 1 2 3 --epochs 3 --amp` | 一键跑多种子（train → test → 汇总），本地用 |
| `python scripts\run_multiseed.py --tag jwfd --seeds 1 2 3 4 5 --epochs 3` | 同上，**P40 上不要加 `--amp`** |
| `python scripts\run_multiseed.py ... --resume` | 断点续跑：已有 checkpoint 的种子跳过训练 |
| `python scripts\aggregate_seeds.py --pattern "jwfd_seed*"` | 单独运行多种子汇总（mean ± std） |

> `run_multiseed.py` 内部调用 `main.py test` 时**强制带 `--val-threshold`**，
> 保证阈值协议被遵守、不会泄漏。

### 4.5 一条龙与工具

| 命令 | 作用 |
|---|---|
| `python main.py all --tag jwfd --val-threshold` | check-data → train → test |
| `python scripts\prepare_jwfd.py` | 离线预缩 600→224（需再改 `USE_PREPROCESSED_PNG=True`） |
| `python scripts\inspect_jwfd.py` | 数据集探索与统计 |



## 五、数据

数据集：**JW-FD（JinWu-Flare Dataset）v1**，约 10% 代表性子集

- 论文：[JW-FD: A 15-Year Multimodal Dataset for Solar Flare Forecasting](https://doi.org/10.3390/universe12090281)（*Universe* 2026, 12(9), 281）
- 代码：<https://github.com/Xiaoxuan-1/JW-FD>
- 数据：<https://doi.org/10.5281/zenodo.21672850>
- 内容：**307 个活动区 / 200,938 张 600×600 单通道磁图 PNG** + **29 个磁场特征**
  （梯度 7、中性线 13、小波 5、磁通量 4）+ **28 个二分类标签 + 7 个类别标签**
- 采样节奏：约 12 分钟（HMI 原生）
- 划分：**活动区级 8:1:1**（`split_dataset.py`，seed=62）
  - train 158,935 / val 18,262 / test 23,741
- 主任务标签：`flare_label_M1.0_48hr`（观测后 48h 内发生 ≥M 级耀斑）

**目录结构**

```
data/raw/
├── png_600_Th1000/{活动区号}/AR*_*.png      600×600 单通道 PNG（307 个子目录）
├── flare_labels.csv                        1,306 条耀斑事件表
└── split_dataset.py                        官方划分脚本
data/index/
├── solar_flare_dataset_png_Lat60_Lon60_Th1000.csv         全量 200,938 行
├── solar_flare_dataset_png_Lat60_Lon60_Th1000_train.csv   158,935 行
├── solar_flare_dataset_png_Lat60_Lon60_Th1000_val.csv      18,262 行
└── solar_flare_dataset_png_Lat60_Lon60_Th1000_test.csv     23,741 行
```

### ⚠️ 关于 v1 与 v2 的差异（影响可比性）

我们手上是 Zenodo **v1（10% 子集）**，而论文描述的是 **v2（完整 15 年、199 万样本）**：

| | v1（本项目） | v2（论文） |
|---|---|---|
| 规模 | 200,938 样本 | 1,991,247 样本 |
| 划分种子 | **seed = 62** | seed = 3970 |
| PNG 饱和度档位 | 只有 **Th1000** | 十档 200–2000 G（推荐 800） |
| 冻结划分名单 | 无 | `split_v2_ar_membership.json` |

**因此本项目的结果不能与论文报告的 v2 基线直接比较，也不能与原论文的 TSS 0.8251 直接比较**——
必须在本文的设定（v1 + seed62 + Th1000）下**自建基线**。

### 关于任务口径

JW-FD 的标签定义是 `t ∈ [Begin−h, Begin)`，解出来即"**耀斑将在观测后 h 小时内爆发**"，
与原论文 SOLAR-STORM2 的"48 hours starting at the observation"在**语义上属于同一类任务**。
差别主要在数据集本身（活动区集合、时段、特征体系、划分方式、版本），
而不是任务定义方向。


## 六、数据划分：活动区级 8:1:1（官方给定）

JW-FD 官方用 `split_dataset.py` 按**活动区（Active Region）**划分，不是按时间：

| | 活动区数 | 样本数 | M1.0@48h 正样本 | 正样本率 |
|---|---|---|---|---|
| 训练集 | 245 | 158,935 | 10,834 | 6.82% |
| 验证集 | 31 | 18,262 | ~1,246 | ~6.8% |
| 测试集 | 31 | 23,741 | 2,087 | 8.79% |
| **合计** | **307** | **200,938** | — | — |

**三组活动区两两零重叠**（`main.py check-data` 会验证这一点）。

### 为什么这个划分比时间划分更严格

时间划分下，同一个活动区的不同时刻可能同时出现在训练集和测试集里 ——
模型只要"见过这个活动区长什么样"，就能靠它拿分，**这不是真正的泛化**。

活动区级划分保证**测试集的活动区长什么样，模型完全没见过**。所以指标会更低，
但更诚实。本项目 TSS 明显低于论文的 0.8251，**有相当一部分要归因于这个更严格的划分**。

> 旧版本（SOLAR-STORM2 时代）用的是时间划分 + 负样本下采样，
> 相关记录保留在 [docs/notes/dataset_build.md](docs/notes/dataset_build.md)。


## 七、实测结果（JW-FD v1 基线）

> ⚠️ **与论文的 TSS 0.8251 不可直接比较**：数据集（JW-FD v1 vs SOLAR-STORM2）、
> 特征体系（29 vs 10）、划分方式（活动区级 vs 时间）、数据版本全部不同。
> 这里的数字是**在本项目设定下自建的基线**。

### 7.1 加权采样器对照（同一份代码，仅采样器不同）

| 指标 | 加权采样（旧默认） | **不加权（现默认）** |
|---|---|---|
| 实验 tag | `jwfd_seed1` | `jwfd_nows` |
| 训练轮数 | 13（早停） | 3 |
| 验证集 AUC | 0.8712 | **0.8966** |
| **测试集 AUC** | 0.8567 | **0.8803** |
| **测试集 TSS** | 0.5893 | **0.6545** |
| TPR | 0.9171 | **0.9554** |
| TNR | 0.6722 | 0.6990 |
| FAR | 0.3278 | 0.3010 |
| F1 | 0.3449 | 0.3763 |
| 验证集选定阈值 | 0.10 | **0.01** |

**结论：`USE_WEIGHTED_SAMPLER` 已改为 `False`。**
原因见 [docs/reports/JW-FD迁移与代码改动总账.md](docs/reports/JW-FD迁移与代码改动总账.md)：
在 JW-FD 上（正样本 6.8%）加权采样会把正样本过采样约 14 倍，导致模型记忆正样本、
AUC 反而下降；论文那套「先负样本下采样到 1:1.8，再加权采样」在 JW-FD 上并不适用。

### 7.2 训练动态：模型第 1 轮就饱和

`jwfd_nows` 的三轮验证 AUC：**0.8966 → 0.8827 → 0.8782**（一路下降），
而训练准确率从 0.934 涨到 0.942。

**这是过拟合**：27.8M 参数 vs 158,935 训练样本，第 1 轮之后泛化能力不再提升。
所以基准线**用 3 个 epoch 就够**，多训无益。

### 7.3 阈值必须靠验证集选

不加权后模型输出概率整体偏低，固定阈值 0.5 下验证集 **TPR 仅 0.014**（几乎全预测负类）。
所以必须用 `--val-threshold`：

```powershell
python main.py test --tag jwfd_nows --val-threshold
```

验证集扫描结果（尖峰，不是平台）：

| 阈值 | 0.005 | 0.010 | 0.015 | 0.020 | 0.030 | 0.050 | 0.500 |
|---|---|---|---|---|---|---|---|
| 验证 TSS | 0.6276 | **0.6707** | 0.6678 | 0.6504 | 0.6294 | 0.5918 | -0.0012 |
| 测试 TSS | 0.5818 | 0.6545 | 0.6677 | 0.6612 | 0.6330 | 0.5996 | 0.1455 |

> 注意：测试集在 0.015 处更高（0.6677），但**那是看测试集才知道的，不能用于报告**。
> 诚实报告值 = 验证集选出的 0.010 对应的 **TSS = 0.6545**。



### 7.4 历史参照：SOLAR-STORM2 时代的数字（**仅存档，不可与上面比较**）

在切换到 JW-FD 之前，本项目在 SOLAR-STORM2 上跑过一套完整实验，记录如下：

| 实验 | TSS | AUC | 说明 |
|---|---|---|---|
| 5 种子（50 epoch 上限，早停） | **0.8321 ± 0.0231** | **0.9664 ± 0.0029** | `experiments/reports/seed_summary.md` |
| 论文 STCNet（单次） | 0.8251 | — | 待复现，见下 |

⚠️ **这组数字与第七节 7.1 的 JW-FD 结果不可比**，因为：
数据集不同、特征不同（10 vs 29）、划分方式不同（时间 vs 活动区）、数据版本不同。
把它单列在此，是为了将来做「跨数据集泛化性对比」时还能找到原始记录。

> 数据已从项目中清理以节省空间。若需复跑，把 SOLAR-STORM2 数据集放回
> `data/raw/magnetogram_jpg/` + `data/raw/para_flare.txt`，
> 并参考 `src/data/preprocess.py`（已标记作废但保留了完整逻辑）与
> [docs/notes/dataset_build.md](docs/notes/dataset_build.md)。


## 八、每个 .pth 权重文件装了什么

**预训练权重（`pretrained/` 下，需要拷到新机器）**

| 文件 | 大小 | 内容 | 用途 |
|---|---|---|---|
| `swin_tiny_patch4_window7_224.pth` | 109 MB | `{'model': OrderedDict(190 项)}`，`patch_embed=(96,1,4,4)`，Swin-Tiny **单通道**预训练 | ✅ **使用**，初始化主干（183/185 参数加载） |
| `cnn_premodel.pth` | 0.4 MB | `OrderedDict(16 项)`，但形状是 **Conv2d** 的 `(16,1,1,1)` | ❌ 不用（形状不匹配 `Conv1d`） |

**训练产物（`experiments/checkpoints/` 下，均已 `.gitignore`，不用拷）**

| 文件 | 大小 | 说明 |
|---|---|---|
| `jwfd_nows_best.pth` | 106.5 MB | **当前最好基线**（不加权采样，3 epoch，AUC 0.8803） |
| `jwfd_seed1_best.pth` | 106.5 MB | 加权采样对照组（13 epoch，AUC 0.8567） |
| `jwfd_base_best.pth` / `jwfd_best.pth` / `amp_ab_best.pth` | 各 106.5 MB | 早期过程性实验（2 epoch），可删 |
| `stcnet_best.pth` / `stcnet_seed1~5_best.pth` | 各 105.7 MB | **SOLAR-STORM2 时代**的 5 种子结果（TSS 0.8321±0.0231），存档用 |
| `demo_best.pth` | 105.7 MB | 最初的功能验证，可删 |

**全部是 `state_dict`**（不是完整模型对象），可以被 `torch.load(..., weights_only=False)` 直接读取。

三个关键发现：

1. **预训练权重外层包了一层 `{'model': ...}`**，必须取 `["model"]` 才能加载
   —— 这正是原代码最大的 bug，漏了这一步导致 352 MB 权重**静默失效**（0/185 加载）。
2. **`patch_embed=(96,1,4,4)` 证明学长实际用的是 Swin-Tiny 而非 Base**（Base 是 128 通道），
   本项目据此设置 `SWIN_VARIANT`。
3. 学长原始的 `model-27/29.pth` 是**完整模型对象**（`torch.save(model)`），
   新版 PyTorch 的 `weights_only=True` 默认值会导致加载失败；
   本项目改存 `state_dict`，并在 `main.py`/`trainer.py` 里保留对旧格式的兼容分支。


## 九、改动了什么

完整记录（含每一处 diff、修复原因、实测数据）见
**[docs/reports/JW-FD迁移与代码改动总账.md](docs/reports/JW-FD迁移与代码改动总账.md)**。
这里只给索引。

### 9.1 阶段一：学长原始代码 → 本项目（修 bug）

**详细复现见 [docs/troubleshooting/bugs.md](docs/troubleshooting/bugs.md)**：

| # | 问题 | 严重程度 | 修复位置 |
|---|---|---|---|
| 1 | 预训练权重静默加载失败（0 个参数加载上） | **致命** | `stcnet.py::load_pretrained` |
| 2 | 磁场参数输入 4 维，`Conv1d` 报错 | **致命** | `dataset.py`、`cnn.py` |
| 3 | 主干型号三方不一致（配置写 Base、代码写 Base、权重实际是 Tiny） | 高 | `config.SWIN_VARIANT` |
| 4 | `torch.load` 新版 `weights_only` 报错 | 中 | `trainer.py`、`main.py` |
| 5 | `forward` 里每 batch 重复加载 336 MB 权重 | 中 | `stcnet.py::__init__` |
| 6 | matplotlib 中文标题缺字 | 低 | `visualization.py` |

另外还修了：分组学习率的过滤条件（`'bert'` 残留导致两组重叠）、
重复 softmax（CE 用在概率上）、保存整个模型等问题。

### 9.2 阶段二：SOLAR-STORM2 → JW-FD 迁移

| 文件 | 改动 |
|---|---|
| `config.py` | 数据路径；`NUM_PARAMS 10→29`；新增 29 个特征列名 `FEATURE_COLS` 与 `LABEL_COL`；轮数/耐心值调整 |
| `src/models/cnn.py` | `Linear(32*10, 320)` → `Linear(32*num_params, out_dim)`（**不改必崩**） |
| `src/models/stcnet.py` | 修 `from cnn import CNN` 导入错误；把特征数传进 CNN |
| `src/data/dataset.py` | **重写**：66 列按列名读取 + 图片路径解析 |
| `main.py` | 去掉 73810/10 维硬编码；`build-data` → `check-data` |
| `src/data/preprocess.py` | 标记**已作废**（官方已给划分，不需要自己切分+下采样） |
| `setup_project.py` | 检查项改为 JW-FD 路径 |

### 9.3 阶段三：实验方法学的修正（**比模型改动更重要**）

| 改动 | 原因 |
|---|---|
| **阈值改为"验证集选 → 测试集报告"** | 原 `--sweep` 在测试集上扫，是**数据泄漏**，实测虚高可达 0.04 |
| **早停判据 `tss` → `auc`** | `tss` 依赖固定阈值 0.5，两次运行能差 0.04–0.06；AUC 稳定 8 倍 |
| **加权采样器 `True` → `False`** | 在 JW-FD 上把正样本过采样 14 倍，AUC 反降 0.024 |
| **新增 AMP 混合精度（`--amp`）** | 唯一能在量级上提速的优化，实测 1.78 倍 |
| **新增 `val_AUC` 日志与曲线** | 用阈值无关指标判断"模型在不在进步" |



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

数据集引用（当前使用）：

> **JW-FD: A 15-Year Multimodal Dataset for Solar Flare Forecasting**,
> *Universe* 2026, 12(9), 281.
>
> - 论文：<https://doi.org/10.3390/universe12090281>
> - 代码：<https://github.com/Xiaoxuan-1/JW-FD>
> - 数据：<https://doi.org/10.5281/zenodo.21672850>
>
> ⚠️ 本项目使用的是 Zenodo **v1（10% 代表性子集，split seed=62，仅 Th1000 档）**，
> 与论文描述的 **v2（全量 199 万样本，seed=3970，十档阈值）** 不同，
> 因此**不能与论文报告的指标直接比较**。

数据集引用（历史，SOLAR-STORM2 时代）：

> the dataset of Active-Region magnetogram and flare provided by
> Space Environment Warning and AI Technology Interdisciplinary Innovation
> Working Group [Xin H, et al., 2018]

- Xin H, Huaning W, Long X, et al. Deep Learning Based Solar Flare Forecasting
  Model. I. Results for Line-of-sight Magnetograms[J]. Astrophysical Journal,
  2018, 856(1): 7.
