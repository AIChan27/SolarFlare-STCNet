# ============================================================
# 权重文件的来源、结构与用途说明
# ============================================================
#
# 本文档回答一个问题：项目里每个 .pth 文件到底装了什么？
#
# 所有结论都是实测得到的（用 torch.load 逐个打开检查 type / 顶层键 /
# state_dict 条目数 / 参数量 / patch_embed 形状），不是猜测。
# ============================================================


## 一、总览

原始两个文件夹（`SwinT-cnn/` 与 `SwinT处理图像/`）里一共有 5 类 .pth 文件。
按用途分成三类：

| 类别 | 文件 | 大小 | 用途 |
|---|---|---|---|
| ① 主干预训练权重 | `swin_tiny_patch4_window7_224.pth` | 109 MB | ImageNet 预训练，用来初始化 Swin 主干 |
| ① 主干预训练权重 | `swin_base_patch4_window7_224.pth` | 336 MB | 同上，但本项目实际没用 |
| ② 分支预训练权重 | `cnn_premodel.pth` | 0.4 MB | 磁场参数 1D-CNN 分支的初始化权重 |
| ③ 训练产物 | `model/history/model-27.pth`、`model-29.pth` | 各 105.7 MB | 学长训练出来的**完整模型对象** |

**重复情况**：`SwinT-cnn/` 和 `SwinT处理图像/` 各自存了一份
`swin_tiny_patch4_window7_224.pth` 和 `swin_base_patch4_window7_224.pth`，
两份文件字节完全相同（都是 352,783,401 / 114,337,157 字节），是纯冗余。
本项目只在 `pretrained/` 下保留一份 tiny + 一份 cnn，base 那份没有复制
（因为确认实际训练用不到，见第四节）。


## 二、① Swin 主干预训练权重

### `swin_tiny_patch4_window7_224.pth`（本项目实际使用）

```
文件大小      : 114,337,157 字节 (109.0 MB)
加载后类型    : dict
顶层键        : ['model']                    ← 注意！只有这一个键
内层类型      : OrderedDict
内层条目数    : 190
参数总量      : 28,534,986  (108.9 MB, fp32)
内层前 3 个键 : patch_embed.proj.weight / patch_embed.proj.bias / patch_embed.norm.weight
patch_embed.proj.weight 形状 : (96, 1, 4, 4)
```

关键结论：

- **`(96, 1, 4, 4)` 说明三件事**：
  - `96` = `embed_dim`，即这是 **Swin-Tiny**（Base 是 128）；
  - `1` = 输入通道数为 **1**，说明权重是**专门为单通道灰度磁图改过并重训的**，
    不是原版 ImageNet 三通道权重；
  - `4` = patch 边长 4，与论文"每块包含 4×4 个像素"一致。
- 190 个条目对 185 个模型参数键，多出来的 5 个是 Swin 官方实现注册的
  `attn_mask` 缓冲区，本项目的 `swin.py` 没注册这些缓冲，属正常差异
  （加载时用 `strict=False`，会出现在 `unexpected_keys` 里，可忽略）。
- 少了 `fc.weight` / `fc.bias`，这是**刻意剔除的分类头**：
  官方权重输出 1000 类（ImageNet），本项目要输出 320 维特征给融合头用，
  所以加载前会把 `fc.*` 和 `head.*` 删掉。

### `swin_base_patch4_window7_224.pth`（本项目未使用）

```
文件大小      : 352,783,401 字节 (336.4 MB)
加载后类型    : dict
顶层键        : ['model']
内层条目数    : 364
参数总量      : 88,100,268  (336.1 MB, fp32)
patch_embed.proj.weight 形状 : (128, 1, 4, 4)
```

同样是单通道、patch 4×4，但 `embed_dim = 128`，是 Swin-Base，
参数量是 Tiny 的 3.09 倍。

**为什么不用它**：见第四节——学长实际训练出来的是 Tiny。


## 三、② `cnn_premodel.pth`：磁场参数分支的预训练权重

```
文件大小      : 418,293 字节 (0.4 MB)
加载后类型    : OrderedDict          ← 注意：没有外层 'model' 包裹，直接就是 state_dict
条目数        : 16
参数总量      : 103,490
```

完整的 16 个键：

```
conv1.weight (16, 1, 1, 1)      conv1.bias (16,)
bn1.weight   (16,)              bn1.bias   (16,)
bn1.running_mean (16,)          bn1.running_var (16,)
bn1.num_batches_tracked ()
conv2.weight (32, 16, 1, 1)     conv2.bias (32,)
bn2.weight   (32,)              bn2.bias   (32,)
bn2.running_mean (32,)          bn2.running_var (32,)
bn2.num_batches_tracked ()
fc.weight    (320, 320)         fc.bias    (320,)
```

关键结论：

- 这**正好就是本项目 `src/models/stcnet.py` 里 `CNN` 类的结构**：
  `Conv1d(1→16, k=1)` → `BN` → `Conv1d(16→32, k=1)` → `BN` → `Linear(320→320)`。
  也就是说，它是学长在别处单独训练好、拿来给 1D-CNN 分支做初始化的权重。
- 权重形状是 `(16, 1, 1, 1)`（4 维）而不是 `(16, 1, 1)`（3 维），
  说明**训练它的时候用的是 `nn.Conv2d`（1×1 卷积、输入 `[N,1,10,1]`），
  而不是 `nn.Conv1d`**。
  这反过来解释了原代码 `models.py` 里那个 `CNN2D` 类（定义了但没用到）
  以及 `utils.py` 里 `unsqueeze(1).unsqueeze(2)` 产生 4 维输入的原因
  —— 原代码是从一个 2D-CNN 版本改过来的，改到一半留下了痕迹。
- **本项目默认不加载它**。原因：形状与当前的 `Conv1d` 不匹配
  （`(16,1,1,1)` 塞不进 `Conv1d` 的 `(16,1,1)`），强行加载会报错；
  而且它只有 10 万参数，从头训练完全没问题。
  如果你想让 1D-CNN 分支也吃预训练，需要把它 `squeeze` 成 3 维再加载，
  可以在 `stcnet.py` 的 `CNN` 类里加一个 `load_pretrained` 方法实现。


## 四、③ 训练产物：`model-27.pth` / `model-29.pth`

这是**整个排查过程中最关键的两个文件**，因为它们揭示了学长实际用的模型配置。

```
文件大小      : 110,878,699 / 110,878,507 字节 (各 105.7 MB)
加载后类型    : Mynet                       ← 不是 dict！是完整的模型对象
state_dict 条目数 : 203
参数总量      : 27,636,956
顶层子模块    : ['transformer', 'text_cnn', 'fc_1']
```

因为 `torch.save(model)` 保存的是**整个模型对象**（pickle 序列化），
里面不仅有参数，还有类的定义引用。这带来两个后果（都是坑）：

1. **加载方式的坑**：PyTorch 2.6 起 `torch.load` 的 `weights_only` 默认从
   `False` 改成 `True`，加载这种完整模型对象会直接抛
   `_pickle.UnpicklingError: Weights only load failed ... GLOBAL models.Mynet was not an allowed global`。
   原代码 `train.py` 第 206 行的 `torch.load("./model/model-33.pth")` 在这个环境下必然报错。
   本项目的做法是改用 `torch.save(model.state_dict())`（更规范、更小、
   且 `weights_only=True` 也能加载），同时在 `main.py test` 里保留了对
   旧格式（完整模型对象）的兼容处理。
2. **体积的坑**：存 state_dict 只需 105.7 MB，存整个模型对象也差不多，
   但每个 epoch 存一份会有 50 份，占了 5 GB 以上。

### 从这两个文件读出的关键事实

```
transformer.patch_embed.proj.weight : (96, 1, 4, 4)   → Swin-Tiny！不是 Base！
transformer.fc.weight               : (320, 49)       → 主干输出 320 维特征
fc_1.weight                         : (2, 640)        → 融合头：640 维 → 2 类
text_cnn.conv1.weight               : (16, 1, 1, 1)   → 与 cnn_premodel.pth 完全一致
text_cnn.fc.weight                  : (320, 320)      → 与 cnn_premodel.pth 完全一致
transformer 参数量                  : 27,561,094
text_cnn    参数量                  : 103,490
fc_1        参数量                  : 1,282
```

**推论一：学长实际用的是 Swin-Tiny，不是代码里写死的 Swin-Base。**

原代码 `models.py` 第 47 行硬编码：
```python
self.transformer = swin_base_patch4_window7_224(self.config.transformer_fc)
```
而 `Config.py` 第 17 行写的是 `swin_base_patch4_window12_384`（连 window 和分辨率都不对，
是个从未生效的死字段）。

但实际训练产物的 `patch_embed` 是 `(96, 1, 4, 4)`——只有 Tiny 才有 `embed_dim=96`。
说明学长跑训练时用的那份代码里，主干选的是
`swin_tiny_patch4_window7_224`，同一份代码在传到我们手上时被改动过
（或者两边不同步）。

**这解释了一个重要问题**：`cnn_premodel.pth` 那个 2D 权重、
`CNN2D` 类、4 维输入的 `unsqueeze`——都是 2D 版本的遗迹，
说明代码在 2D/1D、Base/Tiny 之间反复改过。

**推论二：本项目选 Tiny 是对的。**

`config.SWIN_VARIANT = "swin_tiny_patch4_window7_224"` 之后，
`python main.py check` 报出的参数量是 **27,636,956**，
与 `model-27/29.pth` 的 **27,636,956** **完全一致**。
这从数字上证明了配置正确。

**推论三：`transformer.fc.weight` 的形状是 `(320, 49)`。**

`49 = 7 × 7`，是输入 224×224 图像经过 patch_embed（÷4 → 56×56）
再经过 3 次 PatchMerging（÷8）后的特征图大小。
但注意本项目 `swin.py` 里实际走的是 `AdaptiveAvgPool1d(1)`，
所以 `fc` 的输入维度在 forward 中被压成了 49 → 320 的线性层，
这也是为什么删掉 `fc.*` 只影响 ImageNet 分类能力、不影响特征提取。


## 五、本项目 `pretrained/` 目录里放了什么

```
pretrained/
├── swin_tiny_patch4_window7_224.pth   109 MB   从 SwinT-cnn/ 复制
└── cnn_premodel.pth                   0.4 MB   从 SwinT-cnn/ 复制
```

**没有复制** `swin_base_patch4_window7_224.pth`（336 MB），
因为在第四节已经证明实际训练用不到它。
如果你想试 Base 版本，把 `config.SWIN_VARIANT` 改成
`"swin_base_patch4_window7_224"`，再把那个文件复制进来即可；
但要先确认显存够（主干参数从 2756 万涨到 8810 万，约 3.2 倍）。


## 六、原文件夹里那些大文件可以清理吗

可以。以下判断基于"本项目已经把它们该用的部分复制过来了"：

| 文件 | 大小 | 建议 |
|---|---|---|
| `SwinT-cnn/swin_tiny_patch4_window7_224.pth` | 109 MB | 可删（已复制到本项目） |
| `SwinT-cnn/swin_base_patch4_window7_224.pth` | 336 MB | 可删（确认未使用） |
| `SwinT-cnn/cnn_premodel.pth` | 0.4 MB | 可删（已复制） |
| `SwinT处理图像/swin_tiny_patch4_window7_224.pth` | 109 MB | 可删（与上面重复） |
| `SwinT处理图像/swin_base_patch4_window7_224.pth` | 336 MB | 可删（与上面重复） |
| `SwinT-cnn/model/history/model-27.pth` | 105.7 MB | **建议保留**，这是唯一的训练产物证据 |
| `SwinT-cnn/model/history/model-29.pth` | 105.7 MB | **建议保留** |

合计可释放约 890 MB（不含那两个训练产物）。
