# ============================================================
# 原代码 3 个 bug 的定位、复现与修复
# ============================================================
#
# 本文档记录从学长代码（SwinT-cnn/）迁移到本项目时修掉的每一个 bug，
# 包括：现象 → 根因 → 实测证据 → 修复方式 → 对应文件位置。
#
# 所有 bug 都是实测复现过的，不是"看起来有问题"。
# ============================================================


## 概要

| # | 问题 | 严重程度 | 后果 |
|---|---|---|---|
| 1 | 预训练权重静默加载失败 | **致命** | 352 MB 权重白下，主干全程随机初始化 |
| 2 | 磁场参数输入形状错误 | **致命** | 训练第一步就崩，代码完全跑不起来 |
| 3 | 主干型号与配置/产物不一致 | 高 | 参数量差 3.2 倍，与已有权重不兼容 |
| 4 | `torch.load` 新版默认值 | 中 | 加载旧权重直接抛异常 |
| 5 | 每个 batch 重复加载权重 | 中 | 即使修好①，训练也慢到不可用 |
| 6 | 中文标题缺字 | 低 | 图能出，但中文显示为方块 |


## 一、bug 1：预训练权重静默加载失败（最隐蔽）

### 现象

代码能跑通、不报错，但训练出来的模型和"随机初始化"没区别。
`models.py` 第 61–62 行：

```python
pretrained_weights = torch.load('swin_base_patch4_window7_224.pth')
self.transformer.load_state_dict(pretrained_weights, False)   # strict=False
```

### 根因

打开的权重文件顶层结构是：

```python
{'model': OrderedDict([('patch_embed.proj.weight', ...), ...])}
```

真正的 `state_dict` 藏在 **`'model'` 键里面**，外面还包了一层字典。

把整个 `{'model': ...}` 当成 `state_dict` 传给 `load_state_dict`，
它要找的键是 `patch_embed.proj.weight`，实际给的键是 `model`，
**一个都对不上**。而因为传了 `strict=False`，PyTorch 认为
"多余的键可以忽略"，于是不报错、不告警，**静默地什么都不加载**。

### 实测证据

```python
before = model.transformer.patch_embed.proj.weight.clone()
model.transformer.load_state_dict(sd, False)          # 复现原代码写法
after = model.transformer.patch_embed.proj.weight
print(not torch.equal(before, after))                 # 输出 False
```

`False` 意味着装载前后权重**完全没变**，即 0 个参数被加载。

### 修复

```python
# 取内层 state_dict，并剔除 ImageNet 分类头
state = torch.load(path, map_location="cpu", weights_only=False)
weights_dict = state["model"] if "model" in state else state
for k in list(weights_dict.keys()):
    if k.startswith("head.") or k.startswith("fc."):
        del weights_dict[k]
msg = self.transformer.load_state_dict(weights_dict, strict=False)
```

修复后的实测输出：

```
[权重] 主干参数 183/185 个已加载
[权重] 未加载的键（正常，通常是分类头）: ['fc.weight', 'fc.bias']
[权重] 权重里多余的键 5 个（通常为 attn_mask 缓冲，可忽略）
```

**对照参考**：学长另一份代码 `SwinT处理图像/train.py` 第 66 行写的是
`torch.load(args.weights, map_location=device)["model"]` —— **带了 `["model"]`**，
是对的。说明这个 bug 是在做多模态融合版时漏掉的，不是不懂。

### 为什么这个 bug 特别危险

- 没有任何报错、没有任何警告，训练 loss 照样下降（因为主干仍能学，只是没有预训练优势）；
- 论文的核心卖点之一就是"用 ImageNet 预训练的 Swin 做迁移学习"，
  这个 bug 让该卖点完全失效，但表面上看不出来；
- 只有对比"加载前/后权重是否变化"才能发现。

**修复位置**：`src/models/stcnet.py` → `Mynet.load_pretrained()`


## 二、bug 2：磁场参数输入形状错误（训练直接崩）

### 现象

训练第一个 batch 就抛异常：

```
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d,
              but got input of size: [2, 1, 10, 1]
```

### 根因

数据管线和模型对形状的约定不一致：

```python
# utils.py 第 30 行：产生 4 维张量 [N, 1, 10, 1]
P_train_tensor = torch.tensor(P_train_normalized, dtype=torch.float32)\
                      .unsqueeze(1).unsqueeze(2)

# models.py 第 34 行：以为能压掉多余维度
x = x.squeeze(2)     # ← 无效操作
```

`x` 的形状是 `[B, 1, 10, 1]`，`squeeze(2)` 想压掉第 2 维，
但**第 2 维大小是 10，不是 1**，`squeeze` 只在维度大小为 1 时才生效，
所以这一步是空操作，张量仍是 4 维，直接送进 `nn.Conv1d` → 崩。

### 为什么会有这个形状

看 `src/models/transformer_model.py` 里那个被定义却从未使用的 `CNN2D` 类
（用 `nn.Conv2d`、注释写"1x1 卷积核"），以及 `cnn_premodel.pth` 里
`conv1.weight` 的形状 `(16, 1, 1, 1)`（4 维 Conv2d 权重）——
可以确定：**原代码最早是 2D-CNN 版本（输入 `[N,1,10,1]`），
后来改成 1D-CNN 时，只改了模型没改数据管线**，留下了这个 4 维输入。

### 修复

统一成 `[B, 1, 10]`（3 维，符合 `Conv1d` 的 `[N, C, L]` 约定）：

```python
# src/data/dataset.py：只加一维
params = torch.from_numpy(self.params[idx]).float().unsqueeze(0)   # (1, 10)

# src/models/stcnet.py：删掉无效的 squeeze
def forward(self, x):        # x: (B, 1, 10)
    x = self.relu(self.bn1(self.conv1(x)))
    ...
```

`32 通道 × 长度 10 = 320`，与 `nn.Linear(32 * 10, 320)` 严丝合缝，
证明 `[B,1,10]` 才是原本设计的形状。

**修复位置**：`src/data/dataset.py` → `SolarFlareDataset.__getitem__`；
`src/models/stcnet.py` → `CNN.forward`


## 三、bug 3：主干型号三方不一致

### 现象

三个地方对"Swin 用哪个型号"的说法**互不相同**：

| 位置 | 写法 | 实际含义 |
|---|---|---|
| `Config.py` 第 17 行 | `swin_base_patch4_window12_384` | Base + window12 + 384×384 |
| `models.py` 第 47 行 | `swin_base_patch4_window7_224` | Base + window7 + 224×224 |
| `model-29.pth` 实际权重 | `patch_embed.proj.weight = (96,1,4,4)` | **Tiny** + 单通道 + patch4 |

`embed_dim=96` 是 Tiny 独有的特征（Base 是 128），
所以**训练产物证明实际用的是 Tiny**，前两处写法都没生效。

### 后果

- 用 Base 的架构去加载 Tiny 的权重 → 形状不匹配，加载失败；
- 用 Base 训练 → 主干参数 8810 万，是 Tiny 的 3.2 倍，显存和速度都吃紧；
- 无法复用已有的 `model-27/29.pth`（它们是 Tiny）。

### 修复

在配置里显式声明，模型按配置构建：

```python
# config.py
SWIN_VARIANT = "swin_tiny_patch4_window7_224"

# src/models/stcnet.py
variant = getattr(config, "swin_variant", "swin_tiny_patch4_window7_224")
self.transformer = globals()[variant](config.swin_feature_dim)
```

**验证**：修复后 `python main.py check` 报出参数量 **27,636,956**，
与 `model-27/29.pth` 的参数量**完全一致**。

**修复位置**：`config.py` → `SWIN_VARIANT`；`src/models/stcnet.py` → `Mynet.__init__`


## 四、bug 4：`torch.load` 新版默认值

### 现象

```python
model = torch.load("./model/model-33.pth")
```

在新版 PyTorch 上抛：

```
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL models.Mynet was not an
allowed global by default.
```

### 根因

PyTorch 2.6 起把 `torch.load` 的 `weights_only` 默认值从 `False` 改成 `True`。
而原代码用 `torch.save(model)` 保存的是**完整模型对象**（pickle），
不是纯张量字典，所以被拒绝加载。

本机实测环境是 PyTorch 2.14.1，必然命中。

### 另外

`train.py` 第 206 行硬编码加载 `./model/model-33.pth`，
但目录里只有 `model-27.pth` 和 `model-29.pth`，**这个文件根本不存在**，
即使绕过了 `weights_only` 也会 `FileNotFoundError`。

### 修复

两条路都做：

1. **保存时改用 state_dict**（推荐做法）：
   ```python
   torch.save(model.state_dict(), ckpt_path)
   ```
   体积不变，但 `weights_only=True` 也能加载，且不依赖类的定义。

2. **加载时显式放开 + 兼容旧格式**：
   ```python
   state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
   if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
       state = state.state_dict()      # 是完整模型对象，提取 state_dict
   ```

**修复位置**：`src/train/trainer.py` → `train()`；
`main.py` → `cmd_test()`


## 五、bug 5：每个 batch 重复加载 336 MB 权重

### 现象

```python
def forward(self, inx):
    ...
    pretrained_weights = torch.load('swin_base_patch4_window7_224.pth')   # ← 在 forward 里！
    self.transformer.load_state_dict(pretrained_weights, False)
    img = self.transformer(img)
```

`torch.load` 写在 `forward` 里，意味着**每前向一次就读盘 + 反序列化一次 336 MB**。

### 后果

按训练集 6700 条、batch_size 32 计算，一个 epoch 有 210 个 batch，
就要读盘 210 次 × 336 MB ≈ **70 GB I/O**，纯属浪费，
训练时间会被 I/O 拖垮（即使 SSD 也慢一个数量级）。

### 修复

挪到 `__init__`，只在构造模型时加载一次：

```python
def __init__(self, config, pretrained_path=None):
    ...
    if pretrained_path is not None:
        self.load_pretrained(pretrained_path)     # 只在这里加载

def forward(self, inx):
    img = self.transformer(inx[0])                # forward 里只做前向
```

**修复位置**：`src/models/stcnet.py` → `Mynet.__init__`


## 六、bug 6：中文标题缺字

### 现象

画图时刷屏警告：

```
UserWarning: Glyph 26159 (\N{CJK UNIFIED IDEOGRAPH-662F}) missing from font(s) DejaVu Sans.
```

图上所有中文变成空白方块（□□□□）。

### 根因

matplotlib 默认字体 `DejaVu Sans` 不含 CJK 字形。

### 修复

在 `src/utils/visualization.py` 顶部按优先级挑一个系统已有的中文字体：

```python
_CN_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                       "Source Han Sans SC", "PingFang SC"]
for _f in _CN_FONT_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False    # 中文环境下负号会变方块
```

**修复位置**：`src/utils/visualization.py`


## 七、顺带修正的两个"非 bug 但影响结果"的问题

### ① 验证集不能从训练段 B 里切

原计划从论文训练段 B（2017-09 ~ 2019-01）的尾部切验证集，
但实测发现：**该段的 49 个正样本全部集中在 2017-09 这一个月**，
2017-10 之后一个正样本都没有（太阳活动极小期）。

结果切出的验证集是 **0 正 / 187 负**，单类别数据集算不出 TSS 和 F1，
早停完全失效（`val_TSS=0.0000`、`val_FAR=1.0000`）。

**修复**：改用训练段 A 的尾部（2015 年全年）作验证集：
验证集 正 483 / 负 725（约 1:1.5，与论文训练集比例接近），
训练集正样本从 2841 降到 2358（仅损失 17%），
且训练（2010-05~2014-12）与验证（2015-01~2015-09）在时间上不重叠。
详见 `config.py` 的 `VAL_PERIODS` 注释。

### ② 冒烟测试的截断必须分层

`max_train_samples` 最初写成直接取前 N 条，
但索引 CSV 里**正样本全部排在前面**（`preprocess.py` 先写正样本再写负样本），
于是 200 条全是正样本：

```
[SAMPLER] 类别分布: 负=0, 正=200      ← 权重变成 1/0，除零
```

**修复**：改成按标签分层，正负各取一半。修复后：

```
[SMOKE] 训练集已分层截断为 200 条 (正 100 / 负 100)
[SAMPLER] 类别分布: 负=100, 正=100
```
