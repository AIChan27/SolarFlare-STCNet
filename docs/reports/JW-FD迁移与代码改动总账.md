# JW-FD 迁移与代码改动总账

> **文档位置**：`docs/reports/JW-FD迁移与代码改动总账.md`
> **生成日期**：2026-10-17
> **用途**：完整记录"从学长原始代码 → 当前版本"的所有代码差异、修复的 bug、以及 SOLAR-STORM2 → JW-FD 的迁移细节。
> **配套文档**：
> [`bugs.md`](../troubleshooting/bugs.md)（原始 6 个 bug 的复现与修复）
> ｜[`dataset_build.md`](../notes/dataset_build.md)（SOLAR-STORM2 时代的数据集笔记）
> ｜[`checkpoints.md`](../notes/checkpoints.md)（各 .pth 权重解析）
> ｜[`实验总结与后续计划.md`](实验总结与后续计划.md)（5 种子实验结果与后续规划）

---

## 目录

- [0. 本文档的基准与范围](#0-本文档的基准与范围)
- [1. 改动总览](#1-改动总览)
- [2. 阶段一：原始代码 → 重构版（修掉 6 个问题）](#2-阶段一原始代码--重构版修掉-6-个问题)
- [3. 阶段二：SOLAR-STORM2 → JW-FD 迁移（逐文件）](#3-阶段二solar-storm2--jw-fd-迁移逐文件)
- [4. 600×600 → 224×224 的完整链路](#4-600600--224224-的完整链路)
- [5. 数据格式对照表](#5-数据格式对照表)
- [6. 实测数据与参数依据](#6-实测数据与参数依据)
- [7. 已知限制与未完成项](#7-已知限制与未完成项)
- [8. 验证清单（怎么确认改动生效）](#8-验证清单怎么确认改动生效)

---

## 0. 本文档的基准与范围

### 0.1 三个"基准"要说清楚

| 代号 | 是什么 | 在哪 |
|---|---|---|
| **基准 A** | **学长给的原始代码**（论文作者的实现） | `C:\Personal Files\DeepLearning\SwinT-cnn\`（`Config.py`/`models.py`/`utils.py`/`transformer_model.py`/`train.py`） |
| **基准 B** | **本项目初版**（SOLAR-STORM2 版重构） | 本项目 git 第一次 commit 的状态 |
| **当前** | **JW-FD 版** | 现在的工作区 |

### 0.2 两次大改

| 阶段 | 时间 | 做了什么 | 对应章节 |
|---|---|---|---|
| **阶段一** | 10/15 – 10/16 | 基准 A → 基准 B：修 6 个 bug、重构成 `src/` 结构、补齐数据构建、加多种子实验 | 本文 [第 2 章](#2-阶段一原始代码--重构版修掉-6-个问题) |
| **阶段二** | 10/17 | 基准 B → 当前：数据集从 SOLAR-STORM2 换成 JW-FD，改 8 个代码文件 + 新增 1 个脚本 | 本文 [第 3 章](#3-阶段二solar-storm2--jw-fd-迁移逐文件) |

### 0.3 一句话总结差异

> **阶段一解决的是"代码本身跑不起来、且预训练权重静默失效"**；
> **阶段二解决的是"换数据集后维度、路径、标签、划分方式全都变了"**。

---

## 1. 改动总览

### 1.1 阶段二改动的文件（本次迁移，2026-10-17）

| # | 文件 | 改动类型 | 严重程度 | 一句话说明 |
|---|---|---|---|---|
| 1 | `config.py` | 大改 | 🔴 必须 | 数据路径、`NUM_PARAMS 10→29`、新增 29 个特征列名、标签列、轮数/耐心值 |
| 2 | `src/models/cnn.py` | 小改但关键 | 🔴 **不改必崩** | `Linear(32*10, 320)` → `Linear(32*num_params, out_dim)` |
| 3 | `src/models/stcnet.py` | 小改 | 🔴 **不改必崩** | 修 `from cnn import CNN` 的导入错误；把特征数传进 CNN |
| 4 | `src/data/dataset.py` | **重写** | 🔴 必须 | 从"13 列无表头"改为"66 列按列名读取"，含图片路径解析 |
| 5 | `main.py` | 中改 | 🔴 必须 | `check` 的硬编码（73810、10 维）修正；`build-data` → `check-data` |
| 6 | `src/data/preprocess.py` | 作废 | 🟢 减负 | 官方已给 train/val/test，不需要我们切分+下采样 |
| 7 | `setup_project.py` | 小改 | — | 检查项改成 JW-FD 路径 |
| 8 | `scripts/prepare_jwfd.py` | **新增** | ⚡ 性能 | 离线预缩 600×600 PNG → 224×224 JPEG |
| 9 | `README.md` | 文档 | — | 更新数据说明、命令表、成本预估 |

### 1.2 阶段一改动的文件（历史记录，2026-10-15/16）

| 文件 | 改动 |
|---|---|
| `src/models/swin.py` | 工厂函数 `in_chans` 3→1（单通道）；`torch.meshgrid` 加 `indexing="ij"` |
| `src/models/stcnet.py` | 权重加载取 `sd["model"]`；从 `forward` 挪到 `__init__`；按 config 选主干型号 |
| `src/models/supconloss.py` | 从 `stcnet.py` 拆出（逻辑未改） |
| `src/train/trainer.py` | 分组学习率过滤条件修正；`set_worker_seed()`；改用 `nll_loss(log(p))` 避免重复 softmax；保存 `state_dict` 而非整个模型 |
| `src/train/evaluate.py` | 新建（原代码的指标计算散落在 `train.py`） |
| `src/utils/paths.py`、`visualization.py` | 新建 |

---

## 2. 阶段一：原始代码 → 重构版（修掉 6 个问题）

> 完整复现过程见 [`docs/troubleshooting/bugs.md`](../troubleshooting/bugs.md)，这里只摘结论与代码差异。

### 2.1 🔴 bug 1：Swin 预训练权重**静默失效**（最隐蔽）

**基准 A 的写法**（`models.py:61-62`）：

```python
pretrained_weights = torch.load('swin_base_patch4_window7_224.pth')
self.transformer.load_state_dict(pretrained_weights, False)   # strict=False
```

**根因**：权重文件内部结构是

```python
{'model': OrderedDict([('patch_embed.proj.weight', ...), ...])}   # 190 个条目
```

真正的 `state_dict` 藏在 **`'model'` 键里面**。把整个 `{'model': ...}` 当成 `state_dict` 传进去，
PyTorch 要找的键是一个都对不上，而 `strict=False` 让"键全不匹配"也不报错 → **一个参数都没加载**。

**实测证据**：装载前后 `patch_embed.proj.weight` 完全没变（`torch.equal` 返回 `True`）。

**修复后**（`src/models/stcnet.py::load_pretrained`）：

```python
state = torch.load(path, map_location="cpu", weights_only=False)
weights_dict = state["model"] if ("model" in state and isinstance(state["model"], dict)) else state
for k in list(weights_dict.keys()):
    if k.startswith("head.") or k.startswith("fc."):
        del weights_dict[k]          # 剔除 ImageNet 分类头
msg = self.transformer.load_state_dict(weights_dict, strict=False)
```

**实测效果**：`主干参数 183/185 个已加载`（原来是 0/185）。

---

### 2.2 🔴 bug 2：磁场特征输入是 4 维，`Conv1d` 直接崩

**基准 A 的写法**：

```python
# utils.py:30
P_train_tensor = (...).unsqueeze(1).unsqueeze(2)   # 产生 [N, 1, 10, 1]
# models.py:34
x = x.squeeze(2)     # 想压掉多余维度 —— 无效！因为第 2 维大小是 10，不是 1
```

**报错**：

```
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d,
              but got input of size: [B, 1, 10, 1]
```

**根因溯源**：`cnn_premodel.pth` 里 `conv1.weight` 的形状是 `(16, 1, 1, 1)`（**Conv2d 权重**），
加上代码里有个定义了却从未使用的 `CNN2D` 类 —— 说明原代码是**从 2D-CNN 版本改成 1D 时改了一半**。

**修复**：统一成 `[B, 1, N]`（`Conv1d` 要求的 `[N, C, L]`）。

```python
# src/data/dataset.py（当前版）
params = torch.from_numpy(self.params[idx]).float().unsqueeze(0)   # (1, NUM_PARAMS)

# src/models/cnn.py（当前版，已删掉无效的 squeeze）
x = self.relu(self.bn1(self.conv1(x)))    # 输入 (B, 1, N)
```

---

### 2.3 🔴 bug 3：主干型号**三方不一致**

| 位置 | 写的是什么 | 实际含义 |
|---|---|---|
| 基准 A `Config.py:17` | `swin_base_patch4_window12_384` | Base + window12 + 384×384 |
| 基准 A `models.py:47` | `swin_base_patch4_window7_224` | Base + window7 + 224×224 |
| 学长训练产物 `model-29.pth` | `patch_embed.proj.weight = (96, 1, 4, 4)` | **Tiny**（embed_dim=96） + 单通道 |

**`embed_dim=96` 是 Tiny 独有的**（Base 是 128），所以**训练产物证明实际用的是 Tiny**。

**修复**：`config.SWIN_VARIANT = "swin_tiny_patch4_window7_224"`，模型按配置构建。

**验证**：修好后 `main.py check` 报出参数量 **27,636,956**，与 `model-27/29.pth` 的参数量**完全相同**。

---

### 2.4 bug 4：`torch.load` 新版默认值导致加载旧权重报错

PyTorch 2.6 起 `weights_only` 默认从 `False` 改成 `True`，而基准 A 用 `torch.save(model)`
保存的是**完整模型对象**，加载时抛：

```
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL models.Mynet
```

**另外**：`train.py:206` 硬编码 `./model/model-33.pth`，但目录里只有 `model-27.pth` 和 `model-29.pth`，
**这个文件根本不存在**，即使绕过 `weights_only` 也会 `FileNotFoundError`。

**修复（两条都做）**：
1. 保存改用 `torch.save(model.state_dict(), ckpt_path)`；
2. 加载时 `weights_only=False`，并兼容旧格式：

```python
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
    state = state.state_dict()      # 是完整模型对象，提取 state_dict
```

---

### 2.5 bug 5：每个 batch 都重新加载 336 MB 权重

**基准 A**：`torch.load(...)` 写在 `forward()` 里 → 每前向一次就读盘 + 反序列化一次 336 MB。

**修复**：挪到 `__init__`，只在构造模型时加载一次。`forward` 里只做前向。

---

### 2.6 bug 6：matplotlib 中文标题缺字

默认字体 `DejaVu Sans` 不含 CJK 字形，图上中文变空白方块并刷屏警告：

```
UserWarning: Glyph 26159 (\N{CJK UNIFIED IDEOGRAPH-662F}) missing from font(s) DejaVu Sans.
```

**修复**（`src/utils/visualization.py` 顶部）：按优先级挑系统已有中文字体
（`Microsoft YaHei` → `SimHei` → `Noto Sans CJK SC` → …），并设 `axes.unicode_minus = False`。

---

### 2.7 阶段一的其他改进（非 bug，但影响结果）

| 项 | 基准 A 的做法 | 现在的做法 | 为什么 |
|---|---|---|---|
| 分组学习率 | 第三组条件 `'transformer' not in n and 'bert' not in n` 没排除 textcnn，导致 CNN 参数同时出现在两组 | 三个**互斥**条件（`transformer.` / `text_cnn.` / 其余） | 学习率行为不确定 |
| 损失函数 | 模型内部已 `softmax`，又用 `F.cross_entropy(outputs, labels)` → **重复 softmax** | `F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)`，数学上等价于对原始 logits 做交叉熵 | 重复 softmax 会压平梯度 |
| 保存权重 | `torch.save(model)`，每个 epoch 存一个 105 MB 完整对象 | `torch.save(best_state)`（仅最优一次） | 省磁盘、避免 `weights_only` 报错 |
| 验证集 | 无 | 从训练集按时间切出独立验证集 | 训练需要早停依据 |

---

## 3. 阶段二：SOLAR-STORM2 → JW-FD 迁移（逐文件）

> 本章是**本次迁移（2026-10-17）**的完整记录。每节给出"改之前 / 改之后"。

### 3.1 `config.py`

#### 3.1.1 数据目录

**改之前（SOLAR-STORM2）**

```python
DATA_DIR = PROJECT_ROOT / "data"
SOLAR_STORM2_DIR = DATA_DIR / "raw"
MAGNETOGRAM_DIR = SOLAR_STORM2_DIR / "magnetogram_jpg"     # 73,810 张 jpg
PARA_FLARE_TXT = SOLAR_STORM2_DIR / "para_flare.txt"       # 12 列
INDEX_DIR = DATA_DIR / "index"
RAW_JPG_DIR = DATA_DIR / "raw_jpg"
STATS_DIR = DATA_DIR / "stats"
TRAIN_CSV = INDEX_DIR / "trainflare1.csv"                  # 13 列无表头
VAL_CSV   = INDEX_DIR / "validflare1.csv"
TEST_CSV  = INDEX_DIR / "testflare1.csv"
```

**改之后（JW-FD）**

```python
DATA_DIR = PROJECT_ROOT / "data"
JWFD_DIR        = DATA_DIR / "raw"
JWFD_PNG_DIR    = JWFD_DIR / "png_600_Th1000"       # 307 个活动区子目录
JWFD_FLARE_LABELS = JWFD_DIR / "flare_labels.csv"   # 1,306 条耀斑事件
JWFD_SPLIT_SCRIPT = JWFD_DIR / "split_dataset.py"   # 官方划分脚本

# 离线预缩（新增，见 3.8）
JWFD_PNG_224_DIR = DATA_DIR / "processed" / "png_224"
USE_PREPROCESSED_PNG = False

INDEX_DIR = DATA_DIR / "index"
STATS_DIR = DATA_DIR / "stats"
_CSV_STEM = "solar_flare_dataset_png_Lat60_Lon60_Th1000"
TRAIN_CSV = INDEX_DIR / f"{_CSV_STEM}_train.csv"    # 158,935 行
VAL_CSV   = INDEX_DIR / f"{_CSV_STEM}_val.csv"      #  18,262 行
TEST_CSV  = INDEX_DIR / f"{_CSV_STEM}_test.csv"     #  23,741 行
FULL_CSV  = INDEX_DIR / f"{_CSV_STEM}.csv"          # 200,938 行（备用）

# 旧名字保留为别名，防止遗留脚本 import 失败
SOLAR_STORM2_DIR = JWFD_DIR
MAGNETOGRAM_DIR = SOLAR_STORM2_DIR / "magnetogram_jpg"
PARA_FLARE_TXT  = SOLAR_STORM2_DIR / "para_flare.txt"
```

**删除的变量**：`RAW_JPG_DIR`（从未真正使用）。

#### 3.1.2 时间划分 → 标签配置（整段重写）

**改之前**：`TRAIN_PERIODS` / `TEST_PERIODS` / `VAL_PERIODS`（按时间切）、
`DOWNSAMPLE_NEGATIVE` / `DOWNSAMPLE_SEED`（负样本下采样）、`USE_VAL_SPLIT`。

**改之后**：保留这些**名字**（赋空值/False）以免 import 报错，但**实际已弃用**；新增：

```python
LABEL_COL = "flare_label_M1.0_48hr"      # 主任务标签列

POSITIVE_MEANS = "观测时刻之后48小时内，同一活动区将发生M级及以上耀斑"
```

并在注释里写清了 **JW-FD 的标签语义推导**（这点很重要，见 3.1.5）。

#### 3.1.3 特征维度：`NUM_PARAMS` 10 → 29

**改之前**

```python
# 磁场特征参数个数（para_flare.txt 的第 2–11 列）
NUM_PARAMS = 10
```

**改之后**

```python
NUM_PARAMS = 29

FEATURE_COLS = [
    # 梯度类（7）
    "Gradient mean", "Gradient std", "Gradient median", "Gradient min",
    "Gradient max", "Gradient skewness", "Gradient kurtosis",
    # 中性线类（13）
    "NL length", "NL no. fragments", "NL gradient-weighted length",
    "NL curvature mean", "NL curvature std", "NL curvature median",
    "NL curvature min", "NL curvature max",
    "NL bending energy mean", "NL bending energy std",
    "NL bending energy median", "NL bending energy min", "NL bending energy max",
    # 小波类（5）
    "Wavelet Energy L1", "Wavelet Energy L2", "Wavelet Energy L3",
    "Wavelet Energy L4", "Wavelet Energy L5",
    # 磁通量类（4）
    "Total positive flux", "Total negative flux",
    "Total signed flux", "Total unsigned flux",
]

COL_IMAGE_FILENAME = "image_filename"
COL_IMAGE_PATH     = "image_path"

assert len(FEATURE_COLS) == NUM_PARAMS, "特征列数与 NUM_PARAMS 不一致"
```

> **为什么要单独维护一份列名清单**：改之前 `dataset.py` 用 `iloc[:, 1:11]` 按**列号**取数据。
> JW-FD 的 CSV 有 66 列且列顺序不一定固定，按列号取会**静默取错列**。改成按列名取之后，
> 只要改 `FEATURE_COLS` / `LABEL_COL` 就能换特征或换任务，不会错位。

#### 3.1.4 训练轮数 / 耐心值（因为 epoch 概念变了）

| 参数 | 改之前 | 改之后 | 原因 |
|---|---|---|---|
| `NUM_EPOCHS` | 50 | **12** | 训练集从 6,700 → 158,935（24×），一个 epoch 从 46 秒 → 约 18 分钟。50 轮 = 15 小时 |
| `PATIENCE` | 10 | **5** | 每轮 18 分钟，等 10 轮等于白跑 3 小时 |

换算"见过的样本数"：旧 `6,700 × 50 = 335,000`；新 `158,935 × 12 = 1,907,000`（已是旧的 5.7 倍）。

#### 3.1.5 ⚠️ 一个重要的语义澄清（写在 config 注释里）

JW-FD 的官方标签定义是：

$$t \in [\text{Begin}-h,\ \text{Begin})$$

解这个不等式：

$$t \in [\text{Begin}-h,\ \text{Begin}) \iff \text{Begin} \in (t,\ t+h]$$

即：**耀斑将在观测时刻之后的 h 小时内爆发** —— 这与原论文 SOLAR-STORM2 的
"48 hours starting at the observation" **在语义上是同一类任务（未来窗口）**。

所谓"严格爆发前窗口"强调的是：**观测必须早于爆发**（不能用爆发后的帧去"预测"该爆发），
这本来就是预测任务的应有之义。

> **结论**：本项目结果与原论文 0.8251 不可直接比较，**原因在于数据集本身**
> （活动区集合、时段、特征体系、划分方式、以及 **v1 vs v2 版本**），
> **而不是任务定义方向不同**。

---

### 3.2 `src/models/cnn.py`（🔴 不改必崩）

#### 3.2.1 崩溃原因

`Conv1d(kernel_size=3, stride=1, padding=1)` 保持序列长度不变，所以输入 `(B, 1, 29)` 时：

```
Conv1d(1→16, k=3, p=1)  : (B, 1, 29) → (B, 16, 29)
Conv1d(16→32, k=3, p=1) : (B, 16, 29) → (B, 32, 29)
flatten                 : (B, 32, 29) → (B, 928)
```

而 `fc` 仍然只接受 320 维：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x928 and 320x320)
```

#### 3.2.2 改之前

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * 10, 320)          # ← 写死 10 个特征
```

#### 3.2.3 改之后

```python
class CNN(nn.Module):
    def __init__(self, num_params: int = 10, out_dim: int = 320):
        super(CNN, self).__init__()
        if num_params <= 0:
            raise ValueError(f"num_params 必须为正整数，收到 {num_params}")
        self.num_params = num_params
        self.out_dim = out_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * num_params, out_dim)   # N=10→320；N=29→928
```

**`forward` 未改动**（仍然是 conv1→BN→ReLU→conv2→BN→ReLU→flatten→fc）。

#### 3.2.4 参数变化

| | 改之前（N=10） | 改之后（N=29） |
|---|---|---|
| `fc` 输入维度 | 320 | **928** |
| `fc` 参数量 | 320×320 + 320 = **102,720** | 928×320 + 320 = **297,280** |
| 全模型参数量 | 27,638,012 | **27,832,572**（+194,560） |

---

### 3.3 `src/models/stcnet.py`

#### 3.3.1 修导入错误（你重构时引入的）

```python
# 改之前（报 ModuleNotFoundError: No module named 'cnn'）
from cnn import CNN

# 改之后
from src.models.cnn import CNN
```

> 原因：`python main.py` 时 `sys.path[0]` 是**项目根目录**，所以 `import cnn` 会去找
> `<项目根>/cnn.py` —— 而文件实际在 `src/models/cnn.py`。必须写全包路径。

#### 3.3.2 把特征数传进 CNN

**改之前**

```python
self.text_cnn = CNN()
```

**改之后**

```python
num_params = getattr(config, "num_params", 10)      # SOLAR-STORM2=10；JW-FD=29
cnn_out_dim = getattr(config, "cnn_feature_dim", 320)
self.text_cnn = CNN(num_params=num_params, out_dim=cnn_out_dim)
```

> 用 `getattr` 而不是直接 `config.num_params`，是为了兼容没有该字段的旧 Config 对象（默认回落 10）。

---

### 3.4 `src/data/dataset.py`（完全重写）

#### 3.4.1 为什么要重写

| | SOLAR-STORM2 版 | JW-FD 版 |
|---|---|---|
| CSV 表头 | **无表头** | **有表头，66 列** |
| 列含义 | 第1列路径 + 第2–11列参数 + 第12列标签 | 29 特征 + 28 二分类标签 + 7 类别 + 2 元数据 |
| 取列方式 | `iloc[:, 1:11]`（按列号） | **按列名**（`FEATURE_COLS` / `LABEL_COL`） |
| 图片路径 | 直接就是绝对路径 | `'./3947/AR3947_*.png'` 需拼到 `png_600_Th1000/` 下 |
| 图片尺寸 | 尺寸不一（89×57 ~ 892×480） | 统一 600×600 |
| 标签来源 | CSV 最后一列 | **28 个标签列里选一列** |

#### 3.4.2 图片路径解析（新增函数）

```python
def resolve_image_path(rel_path: str, filename: str | None = None) -> str:
    raw = str(rel_path).replace("\\", "/").strip()

    # 从文件名截出 AR 号（'AR3947_xxx.png' → '3947'），供预缩目录使用
    fn = filename if filename else (Path(raw).name or None)
    ar_num = None
    if fn:
        head = str(fn).split("_")[0]
        ar_num = head[2:] if head.upper().startswith("AR") else head

    # 开启预缩时优先用 224×224 JPEG
    if USE_PREPROCESSED_PNG and ar_num and fn:
        return str(JWFD_PNG_224_DIR / ar_num / f"{Path(str(fn)).stem}.jpg")

    if raw.startswith("/") or (len(raw) > 2 and raw[1] == ":"):
        return str(Path(raw))            # 绝对路径直接用

    if raw.startswith("./"):
        raw = raw[2:]                    # 注意：不能用 lstrip('./')，见下

    for c in (JWFD_PNG_DIR / raw, JWFD_DIR / raw):
        if Path(c).exists():
            return str(c)

    if fn and ar_num:
        return str(JWFD_PNG_DIR / ar_num / str(fn))   # 兜底

    return str(JWFD_PNG_DIR / raw)
```

**踩过的三个坑**（都写在注释里）：
1. **不能用 `lstrip('./')`**：`lstrip` 是按**字符集**剥离，会把路径开头所有 `.` 和 `/` 都删掉，
   `'./3947/...'` 碰巧没事，但 `'../../x'` 就会出错。正确做法是判断前缀后切片。
2. **活动区目录名是裸数字**：`png_600_Th1000/3947/`，**不是** `AR3947/`。
3. 来源路径可能带反斜杠，先统一成正斜杠。

#### 3.4.3 Dataset 读取逻辑

**改之前**

```python
self.df = pd.read_csv(csv_file, header=None)      # 无表头
if self.df.shape[1] != NUM_PARAMS + 2:
    raise ValueError(...)
self.img_paths = self.df.iloc[:, 0].astype(str).tolist()
self.params   = self.df.iloc[:, 1:1+NUM_PARAMS].to_numpy(dtype=np.float32)
self.labels   = self.df.iloc[:, -1].to_numpy(dtype=np.int64)
```

**改之后**

```python
self.label_col = label_col or LABEL_COL

# 只读需要的列（29 特征 + 1 标签 + 2 元数据），降低内存与读盘
need_cols = list(FEATURE_COLS) + [self.label_col, COL_IMAGE_PATH, COL_IMAGE_FILENAME]
df = pd.read_csv(self.csv_file, usecols=lambda c: c in set(need_cols))

# 列完整性自检（缺列就明确报错，而不是静默出错）
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing: raise ValueError(...)
if self.label_col not in df.columns: raise ValueError(...)

# 图片路径一次性预解析（十几万条也就几十毫秒）
self.img_paths = [resolve_image_path(r, f) for r, f in zip(rels, fns)]

# 特征与标签按名取
self.params = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
self.labels = pd.to_numeric(df[self.label_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
```

**新增 `label_col` 参数**：默认用 `config.LABEL_COL`，也可以在构造时传别的列，
方便后续做"多窗口/多阈值"实验而不用改 config。

**新增 `self.n_rows`**：冒烟测试截断数据时同步更新 `__len__` 的返回值
（旧版截断后 `__len__` 仍返回原长度，是个隐患）。

#### 3.4.4 特征归一化

逻辑与旧版一致（min-max，min/max 只在**训练集**统计，验证/测试复用，避免泄漏），
但改为按列名统计，并打印 29 个特征各自的取值范围：

```python
df = pd.read_csv(csv_file, usecols=lambda c: c in set(FEATURE_COLS))
params = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
param_min = np.nanmin(params, axis=0)
param_max = np.nanmax(params, axis=0)
```

> **为什么必须归一化**：29 个特征量纲差异极大。实测训练集：
> `Wavelet Energy L1` ≈ 9.3e4 ~ 6.6e5，`NL curvature` ≈ −1.6 ~ 4.7，
> `Total unsigned flux` ≈ 3.7e5 ~ 3.0e7。不归一化的话大数量级特征会完全主导梯度。

#### 3.4.5 其余函数

| 函数 | 是否改动 | 说明 |
|---|---|---|
| `build_transform` | 未改逻辑，仅补注释 | JW-FD 的 0 G 正好落在灰阶 127，经 `Normalize(0.5, 0.5)` 后约等于 0，**无需改动**（数学见 [4.4](#44-归一化数学为什么-0-g-落在-0-附近)） |
| `build_weighted_sampler` | 未改逻辑，仅更新注释与打印 | 正样本率从 1:1.7 变成 1:13.7，加权采样更有必要 |
| `get_dataloaders` | 增加 `label_col` 参数 | 其余逻辑不变（含 `worker_init_fn`、`persistent_workers`） |

---

### 3.5 `main.py`

#### 3.5.1 `check` 命令：清掉硬编码

**改之前的 4 处硬编码**

```python
paths = {
    "磁图目录": config.MAGNETOGRAM_DIR,        # 旧数据集
    "参数表":   config.PARA_FLARE_TXT,         # 旧数据集
    ...
}
n_jpg = sum(1 for _ in Path(config.MAGNETOGRAM_DIR).glob("*.jpg"))
flag = "OK " if n_jpg == 73810 else "注意"     # ← 写死 73810
...
_, probs = model((torch.randn(1, 1, 224, 224), torch.randn(1, 1, 10)))   # ← 写死 10 维
```

**改之后**

```python
paths = {
    "JW-FD 根目录":   config.JWFD_DIR,
    "PNG 磁图目录":   config.JWFD_PNG_DIR,
    "耀斑事件表":     config.JWFD_FLARE_LABELS,
    "Swin 预训练权重": config.SWIN_PRETRAINED,
}
ar_dirs = [d for d in Path(config.JWFD_PNG_DIR).iterdir() if d.is_dir()]
n_png = sum(1 for _ in Path(config.JWFD_PNG_DIR).rglob("*.png"))
flag = "OK " if (len(ar_dirs) == 307 and n_png == 200938) else "注意"

_, probs = model((torch.randn(1, 1, 224, 224),
                  torch.randn(1, 1, config.NUM_PARAMS)))     # 跟随 config
```

CSV 检查也从"数行数"改成"**行数 + 预期值比对**"（158,935 / 18,262 / 23,741），
并打印当前主任务标签列与特征维度。

#### 3.5.2 `build-data` → `check-data`（整段替换）

**删除**：

```python
def cmd_build_data(args):
    from src.data.preprocess import run_preprocess_stage
    summary = run_preprocess_stage()
    return 0 if summary["train"] > 0 else 1
```

**新增** `cmd_validate_data(args)`（约 150 行），做 5 项校验：

| # | 校验内容 | 产出 |
|---|---|---|
| 1 | 三份 CSV 是否存在、行数是否与官方一致 | 表格 + 差异提示 |
| 2 | 66 列是否齐全、28 个标签列 / 7 个类别列、29 个特征列、`LABEL_COL` 是否存在 | 缺失项清单 |
| 3 | 主任务 + 4 个常用标签的正负样本数与正样本率 | 表格 |
| 4 | 抽样 N 张验证图片路径真实存在（默认 200，`--sample` 可调） | 存在/缺失计数 + 缺失示例 |
| 5 | **活动区级划分是否真的没有交叉**（从 image_filename 截出 AR 号比对三个 split） | 三组两两重叠数 |

命令行的 `build-data --no-downsample` 参数一并删除，换成 `check-data --sample N`；
`config.DOWNSAMPLE_NEGATIVE` 的写入逻辑也随之删除。

#### 3.5.3 `cmd_all` 的说明文档

流程变为 **check-data → train → test**（原来第一步是 build-data），
docstring 里补充了 JW-FD 规模下的耗时提醒和 `--tag` 提醒（见 3.5.4）。

---

### 3.6 `src/data/preprocess.py`（标记作废）

文件**保留**（历史参考价值），但在最前面加了一大段作废说明：

```
⚠️ 本模块已作废（DEPRECATED）
作废原因：SOLAR-STORM2 时代官方没划分，需要自己切分+下采样+生成三联 CSV；
          JW-FD 官方 split_dataset.py 已按活动区 8:1:1 划分并给出三份 66 列 CSV，
          所以"切分/下采样/生成 CSV"三件事全部不需要了。
替代方案：python main.py check-data
```

**代码本身未改**（仍可 import，只是没人调用它）。

---

### 3.7 `setup_project.py`

**改之前**：检查 `MAGNETOGRAM_DIR`、`PARA_FLARE_TXT`、两个预训练权重。

**改之后**：检查 `JWFD_DIR`、`JWFD_PNG_DIR`、三份 CSV、`SWIN_PRETRAINED`，
并把"下一步提示"改成 `check → check-data → train --smoke → train`，
缺失时给出 JW-FD 的 Zenodo 链接和目录结构说明。

---

### 3.8 `scripts/prepare_jwfd.py`（新增，230 行）

#### 3.8.1 为什么需要（以及收益到底有多大——实测拆解）

模型输入是 224×224，而原图是 600×600。不预缩的话，每读一张都要在 CPU 上做一次 600→224 缩放。

**先把 epoch 的时间构成算清楚**（用 2-epoch 实测数据反推）：

| 构成 | 耗时 | 占比 |
|---|---|---|
| 训练阶段 GPU（4,967 batch × 193 ms） | **约 959 秒（16 分钟）** | **85–89%** |
| 验证阶段 + 其他开销（571 batch） | 115–164 秒（2–3 分钟） | 11–15% |
| **合计** | **约 1,074–1,123 秒（18 分钟）** | 100% |

**结论：epoch 时间主要是 GPU 计算，不是数据加载。**

所以预缩的收益要如实说：

| 收益项 | 幅度 |
|---|---|
| 验证阶段数据加载（600×600 解码 + 缩放） | 能省 1–2 分钟/epoch |
| 训练阶段 | **基本没有**（GPU 是瓶颈，8 个 worker 已经把数据提前备好） |
| 磁盘读盘量 | 单张 120 KB → **实测 7 KB**，降约 94% |
| 磁盘占用（若要保留一份） | 23 GB → **1.34 GB** |

> **诚实结论：预缩对 epoch 时间的提升约 10%（主要在验证阶段），不是"显著加速"。**
> 它更大的价值是**降低磁盘 I/O 和存储占用**。
> 真正要提速，应该走 **AMP 混合精度**（1.5–2×）或减小模型/数据规模。

#### 3.8.2 做了什么

```python
with Image.open(src) as im:
    if im.mode != "L":
        im = im.convert("L")                      # 保证单通道
    if im.size != (224, 224):
        im = im.resize((224, 224), Image.LANCZOS) # ★ 高保真下采样
    im.save(dst, "JPEG", quality=92, optimize=True)
```

| 项 | 值 |
|---|---|
| 插值算法 | **LANCZOS**（比默认 BILINEAR 更锐利，适合缩小） |
| 输出格式 | JPEG，`quality=92`，`optimize=True` |
| 输出路径 | `data/processed/png_224/{AR号}/{原文件名去扩展名}.jpg` |
| 并行 | `multiprocessing.Pool`，默认 8 进程 |
| 断点续跑 | 目标文件已存在且非空则跳过；`--force` 可强制重转 |
| 体积 | 200,938 × 7 KB ≈ **1.34 GB**（原 23 GB，**实测值**） |

参数：`--size`（默认 `config.IMG_SIZE`=224）、`--quality`、`--workers`、`--splits`、`--limit`、`--force`。

#### 3.8.3 ⚠️ 一个需要知道的副作用

开启 `USE_PREPROCESSED_PNG` 后，训练时 `RandomResizedCrop(224)` 是作用在**已经缩到 224 的图**上，
而不是原始 600×600 的图上。两者的增强效果略有不同：

- 未预缩：从 600×600 里随机裁一块（area 8%~100%）再缩到 224 → 视角变化更大、能看到局部细节
- 已预缩：从 224×224 里随机裁一块（area 8%~100%）再缩到 224 → 相当于**再放大**，会略微损失细节

另外 JPEG 有轻微有损压缩。这些都属于"可以接受但需要记录"的差异。

---

### 3.9 `README.md`

更新了：快速开始（check → check-data → 预缩 → 训练 → 测试）、常用命令表、
数据集说明（JW-FD 内容、目录结构、**v1 vs v2 差异表**、任务口径澄清）、
以及 JW-FD 规模下的耗时预估。

---

### 3.10 `src/train/trainer.py` + `config.py` + `main.py`：新增混合精度（AMP）

> 这是迁移之后**额外增加**的一项优化，目的是解决"一个 epoch 18 分钟"的问题。
> **默认关闭**（`USE_AMP = False`），不改变任何已有实验的行为。

#### 3.10.1 为什么只有 AMP 能真正提速

先把 epoch 时间拆开（用 2-epoch 实测反推）：

| 构成 | 耗时 | 占比 |
|---|---|---|
| 训练阶段 GPU（4,967 batch × 193 ms） | 约 959 秒 | **85–89%** |
| 验证阶段 + 开销 | 115–164 秒 | 11–15% |

- **数据加载**已经优化到底（`num_workers=8`，8.6 ms/batch，远小于 GPU 的 193 ms）
- **加大 batch** 没有收益（bs=64 实测 398.6 ms/batch，与 32 几乎线性）
- 所以只剩"让每个 batch 算得更快"这一条路 → **混合精度**

#### 3.10.2 实测对比

> ⚠️ 下表是在**另一个训练正在运行（GPU 99% 占用）时**测的，绝对值被抬高，
> **但相对倍数仍可参考**。基准 427.8 ms/batch 对应的是争用状态；
> 无争用时实际约 193 ms/batch。

| 方案 | 单 batch | 显存 | 相对提速 |
|---|---|---|---|
| ① FP32（默认） | 427.8 ms | 3.60 GB | 1.00× |
| ② AMP + bf16 | 271.7 ms | 2.55 GB | **1.57×** |
| ③ AMP + fp16 | 281.8 ms | 2.55 GB | 1.52× |
| ④ 冻结 Swin 主干（FP32） | 159.3 ms | 0.72 GB | 2.69× |
| ⑤ 冻结主干 + AMP bf16 | 102.8 ms | 0.55 GB | 4.16× |

> ④⑤ 属于**另一种实验设定**（linear probing，只训练参数分支和分类头），
> 不能直接替代微调，但考虑到 D4 实验发现"过拟合在 Epoch 1-3 就出现"，
> **冻结主干很可能是"既快又更稳"的选择**，值得作为对照实验。

#### 3.10.3 改了什么

**`config.py` 新增两个常量**

```python
USE_AMP = False         # 默认关闭：保持与之前实验完全一致的行为
AMP_DTYPE = "bf16"      # "bf16" 或 "fp16"
```

并在 `Config` 类里暴露为 `self.use_amp` / `self.amp_dtype`。

**`src/train/trainer.py::train_one_epoch` 改动**

```python
# 改之前
optimizer.zero_grad()
fea, probs = model((imgs, params))
loss_ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)
if config.use_supcon:
    loss_con = supcon(fea, labels=labels)
    loss = loss_ce + config.lambda_supcon * loss_con
else:
    loss = loss_ce
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# 改之后
use_amp = bool(getattr(config, "use_amp", False))
amp_dtype = torch.float16 if str(getattr(config, "amp_dtype", "bf16")).lower() == "fp16" else torch.bfloat16
use_autocast = use_amp and device.type == "cuda"
use_scaler = use_autocast and amp_dtype == torch.float16       # 只有 fp16 需要 GradScaler
scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
    fea, probs = model((imgs, params))
    loss_ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)

# 对比损失强制 fp32：内部有 exp/log + 两两相似度，半精度易掉精度/NaN
if config.use_supcon:
    with torch.autocast(device_type="cuda", enabled=False):
        loss_con = supcon(fea.float(), labels=labels)
    loss = loss_ce.to(loss_con.dtype) + config.lambda_supcon * loss_con
else:
    loss = loss_ce

if use_scaler:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                          # 裁剪前必须 unscale
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer); scaler.update()
else:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**三个设计要点**：

1. **默认 bf16 而非 fp16**：bf16 与 fp32 指数范围相同，**不需要 GradScaler**，也不易溢出
   （Swin 注意力里有 `exp`）。fp16 路径也实现了，但更易踩坑。
2. **SupConLoss 强制 fp32**：它内部有 `exp`、`log`、两两相似度矩阵，半精度下容易触发
   该模块自带的 NaN 检查而直接抛错。
3. **fp16 时先 `unscale_` 再裁剪**：否则裁剪的是被放大的梯度，等于没裁。

**`main.py` 新增命令行开关**

```powershell
python main.py train --amp                  # 启用 bf16 混合精度
python main.py train --amp --amp-dtype fp16 # 改用 fp16
python main.py all --tag jwfd --amp         # 一条龙 + 混合精度
```

#### 3.10.4 验证结果

三种模式各跑 3 个 batch 全部通过：

| use_amp | dtype | 结果 | 显存 |
|---|---|---|---|
| False | — | OK，loss 1.0465 | 3.60 GB |
| True | bf16 | OK，loss 1.0342 | **2.55 GB** |
| True | fp16 | OK，loss 1.0752 | **2.55 GB** |

> **注意**：`USE_AMP` 默认 `False`，所以**正在运行的实验不受影响**；
> 想用需要在下次运行加 `--amp`。

---

### 3.11 阈值选择：从"测试集扫"改为"验证集选 → 测试集报告"（修数据泄漏）

> 这是本次迁移后**第二个独立改动**，重要性高于任何模型改动。

#### 3.11.1 问题：原来的 `--sweep` 是测试集泄漏

**改之前**（`main.py::cmd_test`）：

```python
metrics = evaluate(model, test_loader, device, threshold=cfg.eval_threshold)
...
if args.sweep:
    sweep_thresholds(model, test_loader, device, thresholds=cfg.threshold_sweep)
```

`sweep_thresholds` 是在**测试集**上扫的。虽然当时只是"打印出来看"，但它诱导你把
"测试集上 TSS 最高的那个阈值"写进报告 —— 那就是**用测试集答案挑分割线**。

**危害有多大**：实测（SOLAR-STORM2 一次扫描）：

| 阈值 | 0.40 | 0.50 | **0.60** | 0.70 |
|---|---|---|---|---|
| TSS | 0.8257 | 0.8432 | **0.8507** | 0.8115 |

在 0.4–0.7 之间摆动 **0.04**，比我们想验证的模型改进（0.01–0.02）还大。
也就是说：**不修这个问题，后续任何"改进"都可能是从测试集偷来的。**

#### 3.11.2 改之后：两阶段协议

```python
# ---------- 4. 确定分类阈值 ----------
chosen_threshold = float(cfg.eval_threshold)      # 默认固定值
use_val_threshold = bool(getattr(args, "val_threshold", False))

if use_val_threshold:
    # ① 在【验证集】上扫阈值，选出 t*
    val_ds = SolarFlareDataset(config.VAL_CSV, ...)
    val_loader = DataLoader(val_ds, ...)
    chosen_threshold, best_row, val_scan = select_threshold(
        model, val_loader, device, metric="tss", thresholds=cfg.threshold_sweep,
    )

# ---------- 5. 测试集评估（只评估一次，阈值已定死）----------
metrics = evaluate(model, test_loader, device, threshold=chosen_threshold)
metrics["threshold_source"] = "val" if use_val_threshold else "fixed"
```

**新增函数** `src/train/evaluate.py::select_threshold()`：

```python
def select_threshold(net, loader, device, metric="tss", thresholds=..., verbose=True):
    """
    ⚠️ loader 必须是【验证集】。传测试集就是泄漏。
    返回 (best_threshold, best_row, all_rows)
    """
    results = sweep_thresholds(net, loader, device, thresholds=thresholds, verbose=False)
    best_row = max(results, key=lambda r: r[metric.lower()])
    return float(best_row["threshold"]), best_row, results
```

**新增命令行参数**

```powershell
python main.py test --tag X --val-threshold                 # 验证集选阈值（推荐）
python main.py test --tag X --val-threshold --threshold-metric f1   # 换判据
python main.py test --tag X                                  # 仍可固定 0.5（兼容旧行为）
```

#### 3.11.3 实测验证（用 2-epoch 的 `jwfd_base_best.pth`）

**验证集扫描（判据 TSS）→ 选中 t\* = 0.30**

| 阈值 | TPR | TNR | **TSS** | FAR |
|---|---|---|---|---|
| 0.20 | 0.9096 | 0.7371 | 0.6467 | 0.2629 |
| **0.30** | 0.8780 | 0.7726 | **0.6506** ← 选中 | 0.2274 |
| 0.40 | 0.8360 | 0.7994 | 0.6353 | 0.2006 |
| 0.50 | 0.7899 | 0.8241 | 0.6140 | 0.1759 |
| 0.60 | 0.7009 | 0.8497 | 0.5506 | 0.1503 |

**测试集用 t\* = 0.30 评估**：

```
TSS = 0.6088   TPR = 0.8395   TNR = 0.7693
FAR = 0.2307   AUC = 0.8532   F1  = 0.3966
```

**⚠️ 关键观察**：

| | TSS |
|---|---|
| 验证集 @0.30 | 0.6506 |
| 测试集 @0.30 | **0.6088** |
| 测试集 @0.50（改之前的报告值） | 0.6103 |

**验证集选出的最优阈值，在测试集上并没有更好（0.6088 vs 0.6103）。**

这恰恰证明了改动的必要性：
- 若在**测试集**上扫，会发现某阈值能给到 **0.65+**，然后报"提升 0.04"
- 但那是从测试集偷来的，换数据就消失
- 老老实实用验证集选，得到 **0.6088** —— **不好看，但是真的**

> **一句话**：这个改动会让后续所有"成绩"普遍低 0.01–0.04，但每个数字都站得住。

#### 3.11.4 `--sweep` 的新定位

保留但明确标注为**诊断用**：

```
[诊断] 以下扫描在【测试集】上进行，仅供观察趋势，
       其中的'最优点'不能用于报告 —— 报告请用上面那个阈值。
```

---

### 3.12 `scripts/run_multiseed.py`（新增）：一键多种子实验

配合 3.11 的阈值协议，把"多种子基线"固化成一条命令，避免漏参数导致结果不可比。

```powershell
# 3 个种子先看趋势（约 4.5 小时）
python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 --epochs 8 --amp

# 一次到位 5 个种子（约 7.5 小时，可过夜）
python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 4 5 --epochs 8 --amp
```

对每个种子依次执行：

```
main.py train --tag <tag> --seed s --epochs N --amp
main.py test  --tag <tag> --seed s --val-threshold      ← 强制带阈值协议
```

最后自动调用 `scripts/aggregate_seeds.py` 生成 `mean ± std` 汇总表。

特性：`--resume` 断点续跑、单个种子失败不中断其余、全部输出写入
`experiments/logs/multiseed_<tag>.log`。

#### 3.12.1 🐛 踩坑记录：子进程输出被块缓冲（2026-10-17）

**现象**：跑 `run_multiseed.py --seeds 1 --epochs 20 --amp` 后，**20 分钟看不到任何 epoch 输出**，
看起来像卡住了。

**诊断**：其实训练完全正常 —— GPU 92%、`jwfd_seed1_best.pth` 在 11 分钟时就已生成
（说明 epoch 1 已跑完）。但日志文件只有 557 字节，只记录了启动信息。

**根因**：Python 的 stdout 在**连接到管道**（非终端）时默认是**块缓冲（8 KB）**，
不是行缓冲。原来的写法只加了 `bufsize=1`，那控制的是父进程读端的行为，
**管不住子进程自己的写缓冲**。每轮打印才 ~150 字节，远填不满 8 KB，
所以全部积压在子进程缓冲区里，要等进程退出才一次性吐出。

**验证**（A/B 对比，子进程每 1 秒打印一行）：

| 配置 | 第1行到达 | 第2行 | 第3行 | 判定 |
|---|---|---|---|---|
| 原写法 | 3.03s | 3.03s | 3.03s | ❌ 全被缓冲 |
| **加 `PYTHONUNBUFFERED=1`** | **0.03s** | **1.03s** | **2.03s** | ✅ 实时流式 |

**修复**（`scripts/run_multiseed.py::run`）：

```python
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"      # ★ 禁止子进程缓冲 stdout
env["PYTHONIOENCODING"] = "utf-8"  # ★ 避免 Windows 下中文乱码
proc = subprocess.Popen(..., bufsize=1, env=env)
```

**教训**：只要用管道捕获子进程输出，就必须给子进程设 `PYTHONUNBUFFERED=1`
（或 `python -u`），否则永远看不到实时进度。

> **对已启动的运行没有数据损失**：进程退出时缓冲区会 flush，
> 父进程仍会把全部输出读进日志文件，只是"没有实时可见性"。

#### 3.12.2 日志文件是 UTF-8，PowerShell 读取要指定编码

日志用 `encoding="utf-8"` 写入。PowerShell 5.1 的 `Get-Content` 默认按 ANSI 读，
会显示成乱码（`澶氱瀛愬疄楠`）。正确读法：

```powershell
Get-Content "experiments\logs\multiseed_jwfd.log" -Tail 40 -Encoding UTF8
```

#### 3.12.3 🐛 三个后续踩坑与修复（2026-10-17 晚）

**坑 1：日志乱码的根因是"子进程用 GBK 输出，父进程按 UTF-8 解码"**

第一次跑完后日志里出现 `[AMP] ��Ͼ��������ã�dtype=bf16` 这种乱码，而且**信息已被吞掉**（变成了替换字符 `�`，无法还原）。

根因链条：

```
子进程（main.py）
   stdout 编码 = 系统默认 = GBK(cp936)        ← Windows 中文环境
        ↓ 中文按 GBK 编码成字节
   管道
        ↓
父进程（run_multiseed.py）
   Popen(..., encoding="utf-8", errors="replace")   ← 按 UTF-8 解码 GBK 字节
        ↓ 解不出来 → 用 � 替换 → 原始信息永久丢失
```

**修复（已生效于新进程）**：

```python
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"      # ★ 强制子进程用 UTF-8 输出
```

**坑 2：日志文件没有 BOM，PowerShell 按 ANSI 读**

即使内容是正确的 UTF-8，PowerShell 5.1 的 `Get-Content` 在**无 BOM** 时会按系统 ANSI(GBK) 解码，于是又变乱码。

**修复**：创建日志时用 `utf-8-sig`（带 BOM），后续追加用普通 `utf-8`：

```python
logfile.write_text("", encoding="utf-8-sig")   # 只在文件开头写一次 BOM
```

这样 PowerShell / 记事本 / VS Code 都能自动识别编码。已对现有
`multiseed_jwfd.log` 做了 BOM 转换，现在直接 `Get-Content` 即可正确显示。

**坑 3：源码里的 emoji 让 `aggregate_seeds.py` 崩溃，导致汇总表没生成**

日志最后出现：

```
File "scripts/aggregate_seeds.py", line 107, in main
    print("\u26a0\ufe0f  ...")
UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'
```

Windows 中文控制台是 GBK，**打印 emoji 会直接抛异常**。而这个 `print` 位于
"写 markdown 文件"之前，所以 `jwfd_seed_summary.md` **根本没生成**，脚本退出码 1。

**修复**：去掉 emoji，改用纯文本提示：

```python
print("[提示] 建议至少 3 个种子（推荐 5 个）才能得到可靠的 std 估计。")
```

**教训**：
1. 任何跨进程传递文本的代码，都要显式指定两端编码（`PYTHONIOENCODING`）
2. Windows 日志文件加 UTF-8 BOM，兼容性最好
3. **源码里不要用 emoji**，Windows 中文控制台会崩

#### 3.12.4 阈值扫描区间扩展（0.1 → 0.02 起）

`jwfd_seed1` 在验证集上选出的最优阈值 **t\* = 0.10，正好卡在原区间 `(0.1…0.9)` 的下边界**，
提示真正的最优点可能更低。于是把 `THRESHOLD_SWEEP` 改为：

```python
THRESHOLD_SWEEP = (
    0.02, 0.05, 0.08, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.60, 0.70, 0.80,
)
```

扩展后重跑验证集扫描，结果证明**0.10 确实是真实最优点**（尖峰），不是边界效应：

| 阈值 | 0.02 | 0.05 | 0.08 | **0.10** | 0.15 | 0.20 | 0.50 |
|---|---|---|---|---|---|---|---|
| 验证 TSS | 0.5857 | 0.6325 | 0.6429 | **0.6449** | 0.6448 | 0.6343 | 0.5333 |

**为什么最优点这么低**：TSS = TPR − FAR 是按**类内比率**算的，
而验证集正样本只占 6.8%。把阈值从 0.5 降到 0.1，TPR 从 0.686 升到 0.913（+0.227），
FAR 只从 0.153 升到 0.268（+0.115），净值 +0.112 —— 所以低阈值对 TSS 更有利。

---

## 4. 600×600 → 224×224 的完整链路

> 这一章把"像素怎么从 600 变到 224"的每一步都写清楚，包括数据源头。

### 4.1 第一步：官方把 FITS 磁图做成 8-bit PNG（这一步不是我们做的）

JW-FD 官方用下面的公式把 Gauss 量纲的磁场值 B 压成 0–255 的灰度：

$$B' = \mathrm{clip}(B,\ -B_{\mathrm{th}},\ +B_{\mathrm{th}})$$

$$I = \left\lfloor \frac{B' + B_{\mathrm{th}}}{2B_{\mathrm{th}}} \times 255 \right\rfloor$$

其中 $B_{\mathrm{th}}$ 是**饱和阈值**（我们这版是 **1000 G**，论文 v2 用 800 G 作工作点）。
NaN 先填 0 G，图像还做了左右镜像（北在上、东在左的惯例）。

**几个推论**（实测已验证）：

| 输入 B | 输出灰阶 I |
|---|---|
| 0 G | **127**（`(0+1000)/2000×255 = 127.5`，向下取整 127） |
| +1000 G 及以上 | 255（纯白） |
| −1000 G 及以下 | 0（纯黑） |

实测：`min=0, max=255, mean=127.31, 中位数=127`，**灰阶 127 占 36.11%** ——
与"0 G 落在 127"完全吻合。（36% 说明磁图里大量像素是弱场背景。）

> ⚠️ **超过 ±1000 G 的强场会被截断**。真实活动区核心可到 2000–3000 G，
> 所以这些区域在 PNG 里都是纯黑/纯白。这是 8-bit 量化的固有代价，
> 也是为什么论文要发十档 $B_{\mathrm{th}}$ 供不同任务选择。

### 4.2 第二步（方案一，默认）：训练时在线缩放

`src/data/dataset.py::build_transform`：

**训练集**

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),      # ① 从 600×600 随机裁一块并缩放到 224
    transforms.RandomHorizontalFlip(),      # ② 随机水平翻转
    transforms.ToTensor(),                  # ③ uint8 [0,255] → float32 [0,1]
    transforms.Normalize([0.5], [0.5]),     # ④ [0,1] → [-1,1]
])
```

`RandomResizedCrop(224)` 的默认行为：
- **裁剪面积**：原图的 8% ~ 100%（即最短边约 204 px 到整张图）
- **长宽比**：3/4 ~ 4/3
- **插值**：默认 **BILINEAR**
- 先随机裁，再缩放到 224×224

**验证/测试集**

```python
transforms.Compose([
    transforms.Resize((224, 224)),          # 整张图确定性缩放
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
```

> `Resize` 的默认插值也是 **BILINEAR**，且新版 torchvision 默认开启 `antialias=True`
> （缩小前先做抗锯齿）。

### 4.3 第三步（方案二，可选）：离线预缩

```powershell
python scripts/prepare_jwfd.py          # 600×600 PNG → 224×224 JPEG
# 然后把 config.py 里 USE_PREPROCESSED_PNG 改成 True
```

用 **PIL + LANCZOS** 一次性缩好存盘。之后训练时 `Resize` 那一步不再起作用
（图已经是 224），只有 `RandomResizedCrop` / `RandomHorizontalFlip` / `ToTensor` / `Normalize` 生效。

### 4.4 归一化数学：为什么 0 G 落在 0 附近

这是**不需要改代码**的关键原因，把数算一遍就清楚：

```
原始灰阶 I        = 127          （0 G 对应的值）
ToTensor 后       = 127 / 255 = 0.49804
Normalize(0.5,0.5) = (0.49804 − 0.5) / 0.5 = −0.00392
```

**−0.0039 几乎就是 0** —— 也就是说"磁场为零"在模型看来约等于 0，物理含义直观。

| 灰阶 | ToTensor | Normalize 后 | 物理含义 |
|---|---|---|---|
| 0 | 0.0000 | **−1.000** | ≤ −1000 G |
| 127 | 0.4980 | **−0.004** | 0 G |
| 255 | 1.0000 | **+1.000** | ≥ +1000 G |

> 如果这里误用了 3 通道写法 `Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])`，会直接报
> `output with shape [1,224,224] doesn't match the broadcast shape [3,224,224]`。

### 4.5 尺寸链路总表

| 阶段 | 张量形状 | dtype / 范围 | 发生在哪 |
|---|---|---|---|
| 磁盘上的 PNG | 600 × 600 × 1 | uint8，`mode='L'`，(0, 255) | 官方发布 |
| `Image.open` | 600 × 600 | PIL Image，`mode='L'` | `dataset.__getitem__` |
| `RandomResizedCrop(224)` / `Resize` | 224 × 224 | PIL Image | `build_transform` |
| `ToTensor()` | **(1, 224, 224)** | float32，(0, 1) | 同上 |
| `Normalize(0.5, 0.5)` | (1, 224, 224) | float32，(−1, 1) | 同上 |
| DataLoader 拼 batch | **(B, 1, 224, 224)** | float32 | `DataLoader` |
| Swin `PatchEmbed`（patch=4） | (B, 96, 56, 56) → token (B, 3136, 96) | — | `swin.py` |
| 4 个 Stage 后 | (B, 49, 768) | — | `swin.py` |
| `AdaptiveAvgPool1d(1)` + `fc(49→320)` | **(B, 320)** | — | `swin.py` |

---

## 5. 数据格式对照表

### 5.1 CSV 结构

| | SOLAR-STORM2 | JW-FD |
|---|---|---|
| 文件名 | `trainflare1.csv` 等（我们自己生成） | `solar_flare_dataset_png_*_train.csv`（官方提供） |
| 表头 | **无** | **有，66 列** |
| 列数 | 13 | 66 |
| 第 1 部分 | 1 列图片路径 | **29 列特征** |
| 第 2 部分 | 10 列 SHARP 参数 | **28 列二分类标签**（7 窗口 × 4 阈值） |
| 第 3 部分 | 1 列标签（0/1） | **7 列类别标签**（GOES 等级字符串） |
| 第 4 部分 | — | **2 列元数据**（`image_filename`、`image_path`） |
| 行数 | 6,700 / 1,208 / 12,110 | 158,935 / 18,262 / 23,741 |

### 5.2 图片与路径

| | SOLAR-STORM2 | JW-FD |
|---|---|---|
| 格式 | JPG | **PNG** |
| 尺寸 | 不一（89×57 ~ 892×480） | **统一 600×600** |
| 通道 | `mode='L'` | `mode='L'` |
| 组织 | **平铺**在一个目录 | **按活动区分子目录** |
| 路径示例 | `.../magnetogram_jpg/hmi.sharp_720s.948.20111018_044800_TAI.magnetogram.fits.jpg` | `png_600_Th1000/1151/AR1151_20110129_000000.png` |
| CSV 中的路径 | 绝对路径 | **相对路径** `./3947/AR3947_*.png` |
| 采样节奏 | 96 分钟 | **约 12 分钟** |
| 灰度含义 | 直接灰度 | 8-bit 定标，0 G = 灰阶 127 |

### 5.3 划分方式

| | SOLAR-STORM2 | JW-FD |
|---|---|---|
| 谁切的 | **我们自己**（`preprocess.py`） | **官方**（`split_dataset.py`） |
| 依据 | 按**时间**（论文 4.1 节） | 按**活动区**（AR 级 8:1:1） |
| 随机种子 | 42（仅用于负样本下采样） | **62**（v2 是 3970） |
| 泄漏风险 | ⚠️ 同一活动区的不同时刻可能跨训练/测试 | ✅ **同一活动区只进一个分区** |
| 负样本处理 | 按论文规则下采样（每天每 AR 只留一条） | **不下采样**（原始比例） |
| 实测 | train 6,700 / val 1,208 / test 12,110 | train 158,935 / val 18,262 / test 23,741 |
| 活动区数 | — | train **245** / val **31** / test **31**，**两两零重叠** |

---

## 6. 实测数据与参数依据

### 6.1 吞吐（JW-FD，RTX 5070 Laptop 12 GB）

| batch_size | 单 batch 耗时 | 显存峰值 | 预估单 epoch |
|---|---|---|---|
| **32**（当前） | 193.1 ms | 3.60 GB | **约 17.0 分钟** |
| 64 | 398.6 ms | 6.80 GB | 约 17.5 分钟 |

> **结论：加大 batch 没有收益**（Swin 主干是算力受限，不是 batch 受限）。
> 显存还有大量余量（3.6 / 12 GB），瓶颈在算力不在显存。

**2-epoch 实际训练实测**：

| epoch | 耗时（含验证） | train loss | train acc | val ACC | val F1 | val TSS | val FAR |
|---|---|---|---|---|---|---|---|
| 1 | 1123 s（18.7 min） | 0.7287 | 0.8258 | 0.8308 | 0.4896 | 0.5227 | 0.1470 |
| 2 | 1074 s（17.9 min） | 0.6711 | 0.8567 | 0.8200 | 0.5153 | 0.6140 | 0.1759 |

### 6.2 DataLoader 并行度（SOLAR-STORM2 数据上实测，结论可迁移）

| num_workers | 单 batch 数据加载 | 相对加速 |
|---|---|---|
| 0 | 98.2 ms | 1.00× |
| 2 | 38.9 ms | 2.53× |
| 4 | 22.0 ms | 4.45× |
| **8**（当前） | **8.6 ms** | **11.43×** |

> 当时 GPU 计算 195 ms/batch，所以 `num_workers=0` 时数据加载占 1/3 时间，
> 表现为 **CPU 90% / GPU 43%**。开到 8 后 GPU 不再等数据。
> 16 逻辑核，取 8 留一半给主进程与系统。

### 6.3 2-epoch 模型的测试集结果（仅验证链路，不代表性能）

```
测试集 23,741 条（正 2,087 / 负 21,654）
TSS = 0.6103    TPR = 0.7863    TNR = 0.8241
FAR = 0.1759    F1  = 0.4354    ACC = 0.8207
AUC = 0.8532    Precision = 0.3010
混淆矩阵：TN=17,844  FP=3,810  FN=446  TP=1,641
```

**怎么解读**：
- 只有 2 个 epoch，远未收敛，**不能与 SOLAR-STORM2 的 0.8321 或论文的 0.8251 比较**
- **AR 级划分让任务明显更难**：测试集的活动区模型完全没见过（时间划分时同一 AR 会跨集）
- **Precision 从 0.11 涨到 0.30**：正样本率从 1.2% 涨到 8.8%，不再是极端不平衡
- **FAR = 0.176 偏高**：加权采样器让模型偏向预测正类，说明阈值 0.5 不是最优

### 6.4 主任务标签分布（训练集 158,935 条）

| 标签列 | 正样本 | 负样本 | 正样本率 |
|---|---|---|---|
| **`flare_label_M1.0_48hr`（主任务）** | **10,834** | 148,101 | **6.82%** |
| `flare_label_C1.0_24hr` | 26,891 | 132,044 | 16.92% |
| `flare_label_M1.0_24hr` | 6,825 | 152,110 | 4.29% |
| `flare_label_M5.0_48hr` | 1,995 | 156,940 | 1.26% |
| `flare_label_X1.0_24hr` | 381 | 158,554 | 0.24% |

---

## 7. 已知限制与未完成项

| # | 项 | 现状 | 建议 |
|---|---|---|---|
| 1 | ~~阈值未做验证集选择~~ | ✅ **已修复**（见 3.11 节） | 现在用 `--val-threshold` 在验证集上选阈值，测试集不再参与阈值搜索 |
| 2 | **未做离线预缩** | `USE_PREPROCESSED_PNG = False` | 跑 `scripts/prepare_jwfd.py` 后开启，可显著缩短 epoch |
| 3 | **只有 2 epoch 的初步结果** | 远未收敛 | 跑正式 12 epoch（含早停） |
| 4 | **未做多种子** | 只有 seed=42 一次 | 数据量大了之后单次方差更小，但正式结论仍建议 3–5 种子 |
| 5 | **`Gradient min` 恒为 0** | PNG 分支的 8-bit 量化地板，该列无信息量 | 如需剔除，从 `config.FEATURE_COLS` 删掉该行即可（`NUM_PARAMS` 会自动断言失败提醒你改） |
| 6 | **旧数据集未清理** | `data/raw/magnetogram_jpg`（7.26 GB）+ `para_flare.txt` 仍在 | 确认新流程稳定后可删除 |
| 7 | **旧索引 CSV 未清理** | `data/index/trainflare1.csv` 等 3 个文件 | 同上 |
| 8 | **`src/data/preprocess.py` 保留但作废** | 仍能 import，逻辑过时 | 想彻底干净可以删掉；保留的价值是将来做 SOLAR-STORM2 对照实验时参考 |
| 9 | **v1 vs v2 版本差异** | 我的是 v1（10% 子集，seed 62，只有 Th1000） | 论文是 v2（全量，seed 3970，十档）。等 v2 发布后可迁移对比 |
| 10 | **无官方基线对照** | 未在 JW-FD 上重跑论文方法做基线 | 要声称"改进"，必须先在**同一划分**上建立基线 |

---

## 8. 验证清单（怎么确认改动生效）

按顺序执行，每步都给了**预期输出**。

### 8.1 环境与模型

```powershell
python main.py check
```

**预期**：
```
[2] 数据（JW-FD v1，约 10% 代表性子集）
    [OK ] JW-FD 根目录  ...
    [OK ] 活动区 307 个，PNG 200,938 张（10% 子集预期 307 / 200,938）
[3] 数据集 CSV ...
    [OK ] 训练集: 158,935 条（预期 158,935）
    [OK ] 验证集:  18,262 条（预期 18,262）
    [OK ] 测试集:  23,741 条（预期 23,741）
    主任务标签列 : flare_label_M1.0_48hr
    特征维度     : 29（config.FEATURE_COLS）
[4] 模型构建自检
    [权重] 主干参数 183/185 个已加载
    [OK ] Mynet 构建成功，参数量 27,832,572
    [OK ] 前向输出形状 (1, 2)
自检结论: 全部就绪，可以开始训练
```

### 8.2 数据校验

```powershell
python main.py check-data
```

**预期关键行**：
```
[OK  ] train: 158,935 行 / 66 列
    二分类标签列数    : 28
    类别标签列数      : 7
    29 个特征列缺失   : 无
    抽样 200 张 → 存在 200，缺失 0
    train∩val   : 0 个重叠  ✓
    train∩test  : 0 个重叠  ✓
    val∩test    : 0 个重叠  ✓
校验结论: 全部通过，可以开始训练
```

### 8.3 冒烟测试

```powershell
python main.py train --smoke
```

**预期**：`[SMOKE] 训练集已分层截断为 200 条 (正 100 / 负 100)`，跑 1 个 epoch 后 exit 0。
（注意：即使冒烟也会过一遍 18,262 条验证集，约 2 分钟。）

### 8.4 正式训练 + 测试

```powershell
python main.py all --tag jwfd --sweep
```

---

## 附录 A：本次迁移涉及的命令变化

| 旧命令 | 新命令 | 说明 |
|---|---|---|
| `python main.py build-data` | **`python main.py check-data`** | 不再构建，只校验 |
| `python main.py build-data --no-downsample` | （删除） | JW-FD 不做负样本下采样 |
| — | **`python scripts/prepare_jwfd.py`** | 新增：离线预缩 |

其余命令（`check` / `train` / `test` / `all` 及各自的 `--tag`/`--seed`/`--smoke`/`--sweep` 等）**均保持不变**。

## 附录 B：配置文件里改了哪些常量（速查）

| 常量 | 旧值 | 新值 |
|---|---|---|
| `NUM_PARAMS` | 10 | **29** |
| `NUM_EPOCHS` | 50 | **12** |
| `PATIENCE` | 10 | **5** |
| `TRAIN_CSV` | `trainflare1.csv` | `solar_flare_dataset_png_Lat60_Lon60_Th1000_train.csv` |
| `VAL_CSV` | `validflare1.csv` | `..._Th1000_val.csv` |
| `TEST_CSV` | `testflare1.csv` | `..._Th1000_test.csv` |
| `DOWNSAMPLE_NEGATIVE` | `True` | `False`（已弃用） |
| `LABEL_COL` | （不存在） | **`"flare_label_M1.0_48hr"`** |
| `FEATURE_COLS` | （不存在） | **29 个特征列名** |
| `JWFD_PNG_DIR` | （不存在） | `data/raw/png_600_Th1000` |
| `USE_PREPROCESSED_PNG` | （不存在） | `False`（改为 True 前需先跑预缩脚本） |
| `USE_AMP` | （不存在） | `False`（命令行 `--amp` 可临时开启） |
| `AMP_DTYPE` | （不存在） | `"bf16"`（可改 `"fp16"`） |

其余（`BATCH_SIZE=32`、`LR_*`、`DROPOUT=0.3`、`USE_SUPCON=True`、`NUM_WORKERS=8`、
`SWIN_VARIANT=tiny`、`IMG_SIZE=224`、`RANDOM_SEED=42`、`EVAL_THRESHOLD=0.5`）**均未改动**。
