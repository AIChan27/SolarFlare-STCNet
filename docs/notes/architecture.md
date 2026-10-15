# ============================================================
# 模型结构与数据流
# ============================================================
#
# 本文档用一条完整的数据流把各个文件串起来，说明：
#   1. 一个样本从磁盘到模型输出的全过程（每个环节的张量形状）
#   2. STCNet 三个模块的对应关系（与论文 3.1 节对齐）
#   3. 损失函数怎么算
#   4. 关键设计选择的理由
# ============================================================


## 一、总体数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│ 磁盘                                                                 │
│   data/index/trainflare1.csv   一行 = 1 张图 + 10 个参数 + 1 个标签    │
│   ../SwinT-cnn/data/magnetogram_jpg/*.jpg   73810 张灰度磁图          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ src/data/dataset.py
                             │ SolarFlareDataset.__getitem__
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 单样本（DataLoader 会拼成 batch）                                     │
│   img    : Tensor (1, 224, 224)   float32   单通道灰度                │
│   params : Tensor (1, 10)         float32   min-max 归一化后           │
│   label  : Tensor ()              int64     0 或 1                    │
│                                                                     │
│   返回形式：((img, params), label)                                    │
│            ↑ 嵌套元组，为了 model((img, params)) 能直接解包            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ 批处理后
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 一个 batch（B = batch_size，默认 32）                                 │
│   imgs   : (B, 1, 224, 224)                                           │
│   params : (B, 1, 10)                                                 │
│   labels : (B,)                                                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ src/models/stcnet.py
                             │ Mynet.forward((imgs, params))
                             ▼
        ┌────────────────────┴────────────────────┐
        │ 第一部分：Swin Transformer               │ 第二部分：1D-CNN
        │ self.transformer(imgs)                  │ self.text_cnn(params)
        │                                         │
        │ (B,1,224,224)                           │ (B,1,10)
        │   ↓ PatchEmbed  patch=4                 │   ↓ Conv1d(1→16, k=1)
        │ (B, 56, 56, 96)                         │ (B,16,10)
        │   ↓ 4 个 BasicLayer（含 PatchMerging）    │   ↓ ReLU + BN
        │ (B, 7, 7, 768)                          │   ↓ Conv1d(16→32, k=1)
        │   ↓ AdaptiveAvgPool1d(1)                │ (B,32,10)
        │ (B, 768) → 等等，见下方说明               │   ↓ flatten
        │   ↓ nn.Linear(49, 320)                  │ (B, 320)
        │ (B, 320)                                │   ↓ Dropout(0.3)
        │                                         │ (B, 320)
        └────────────────────┬────────────────────┘
                             │ torch.cat([...], dim=1)
                             ▼
                    fea : (B, 640)
                             │ self.fc_1
                             ▼
                  logits : (B, 2)
                             │ self.softmax(dim=1)
                             ▼
                  probs  : (B, 2)   两类的概率，和为 1
                             │
                             ▼
              forward 返回 (图像特征 (B,320), probs (B,2))
```


## 二、关于 Swin 输出维度的说明

上面流程图里 `(B, 7, 7, 768)` 到 `(B, 320)` 这一步值得单独解释，
因为 `src/models/swin.py` 里的实现和常规理解略有不同。

```python
def forward(self, x):
    x, H, W = self.patch_embed(x)        # (B, 3136, 96)  ← 56*56=3136 个 token
    for layer in self.layers:
        x, H, W = layer(x, H, W)         # 逐层下采样
    x = self.norm(x)                     # (B, 49, 768)   ← 7*7=49 个 token
    x = self.avgpool(x)                  # AdaptiveAvgPool1d(1) → (B, 768, 1)
    x = x.view(x.size(0), -1)            # (B, 768)
    x = self.fc(x)                       # 等等，fc 是 Linear(49, 320)？
    return x
```

这里有个细节：`self.fc = nn.Linear(49, num_classes)` 的输入维度写的是 **49**，
但 `avgpool` 之后传进来的却是 **768**。这是因为：

- `AdaptiveAvgPool1d(1)` 对 `(B, 49, 768)` 做的是**在 dim=2（768 那一维）上池化**，
  输出是 `(B, 49, 1)`，而不是 `(B, 768, 1)`；
- 所以 `view` 之后是 `(B, 49)`，正好对上 `Linear(49, 320)`。

也就是说，**它把 49 个空间 token 各自压成 1 个数，然后 49 → 320 做线性映射**，
等价于"全局平均池化 + 一个线性层"，但池化轴与常规写法不同。

这个细节不重要（不影响能否跑通），但解释了为什么
`model-29.pth` 里 `transformer.fc.weight` 的形状是 `(320, 49)`。
**如果以后要改主干型号（比如换成 Base），这个 49 不会变**
（因为 224 / 4 / 2³ = 7，7×7=49，与 embed_dim 无关）。


## 三、三个模块与论文的对应关系

论文 3.1 节把 STCNet 分成三部分，对应关系如下：

| 论文描述 | 代码位置 | 说明 |
|---|---|---|
| 第一部分：原始图像特征提取网络（Swin Transformer），对磁图提特征并展开为一维 | `src/models/swin.py` 的 `SwinTransformer` | 输出 `(B, 320)` |
| 第二部分：磁场特征参数提取网络（CNN），对 10 个参数提特征并展开为一维 | `src/models/stcnet.py` 的 `CNN` | 输出 `(B, 320)` |
| 第三部分：分类网络，拼接两个模态特征后全连接分类 | `src/models/stcnet.py` 的 `Mynet.fc_1` | `(B,640)` → `(B,2)` |
| （额外）对比学习辅助损失 | `src/models/stcnet.py` 的 `SupConLoss` | 让同类特征聚拢 |

论文 3.2 节的参数：
> "图像是单通道的灰度图像，图像在经过调整尺寸、随机水平翻转后的尺寸为 (224,224)，
> 每块包含 4×4 个像素，每个像素有一个灰度值，因此展开后的维度为 4×4×1=16。"

对应代码里的三处设置：
- `config.IMG_SIZE = 224`
- `swin.py` 里工厂函数 `in_chans=1`（单通道）
- `swin.py` 里 `patch_size=4`


## 四、损失函数

训练时的总损失（`src/train/trainer.py::train_one_epoch`）：

```
loss = CrossEntropy(probs, labels) + λ · SupConLoss(图像特征, labels)
```

其中 `λ = config.LAMBDA_SUPCON = 1.0`（等价于原代码的 `(CE + SupCon) / 2`）。

### 为什么用 `F.nll_loss(log(probs))` 而不是 `F.cross_entropy`

原代码写的是：

```python
logits = self.softmax(logits)            # 模型内部已经 softmax 了
loss = F.cross_entropy(outputs, labels)  # cross_entropy 内部又会 log_softmax
```

`cross_entropy` 内部做的是 `log_softmax`，所以对已经 softmax 过的概率再做一次，
等于**重复 softmax**——数值上被压平，梯度变小。

本项目改成：

```python
loss_ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)
```

`nll_loss(log(p))` 与 `cross_entropy(z)` 在数学上完全等价
（因为 `log(softmax(z)) = log_softmax(z)`），
所以这个写法既修正了重复 softmax，又不需要改动模型的返回接口
（模型仍返回概率，方便评估阶段直接用）。

### SupConLoss 的作用

太阳耀斑是少数类（训练集正样本 2358 / 负样本 4342，测试集更是 147 / 11963）。
对比学习让同一类别的样本在特征空间里互相靠近，
使少数类的特征更紧凑，从而提升 TSS。

它作用在**图像模态特征** `(B, 320)` 上，而不是融合后的 640 维特征。


## 五、几个关键设计选择的理由

### 1. 为什么 `params` 是 `(B, 1, 10)` 而不是 `(B, 10)`

`nn.Conv1d` 要求输入是 `[N, C, L]`（3 维）。
把 10 个参数看作"长度 10、通道 1"的一维序列，
所以需要显式加一个通道维。

原代码加了**两维**变成 `(B,1,10,1)`，导致 `Conv1d` 报错，
这是 bug 2 的根源，详见 `docs/troubleshooting/bugs.md`。

### 2. 为什么参数要做 min-max 归一化

实测 10 个参数的量纲差异极大：

```
参数  1: [0, 1.344e+04]
参数  2: [0, 5.358e+24]     ← 10^24 量级
参数  3: [0, 3.167e+14]
参数  8: [307.6, 2.768e+04]
参数 10: [0, 80]
```

不归一化的话，`1e24` 量级的参数会完全主导梯度，`0~1` 量级的参数学不到东西。

归一化的 `min`/`max` **只在训练集上统计**，然后复用到验证/测试集
（`src/data/dataset.py::compute_param_range`），
否则会引入测试集信息，属于数据泄漏。

### 3. 为什么图片用 `RandomResizedCrop` 增强

论文原文提到"图像在经过调整尺寸、随机水平翻转后"。
另外磁图的原始尺寸差异很大（89×57 到 892×480），
`RandomResizedCrop` 既做了尺寸统一，又顺带做了尺度增强。

注意 `Normalize([0.5], [0.5])` 里的 list **长度必须是 1**（对应单通道），
写成 `[0.5, 0.5, 0.5]` 会报
`output with shape [1,224,224] doesn't match the broadcast shape [3,224,224]`。

### 4. 为什么用分组学习率

- Swin 主干有 ImageNet 预训练权重，只需小幅微调 → 小学习率 `2e-4`；
- 1D-CNN 分支是随机初始化的，需要从头学 → 大学习率 `1e-3`。

原代码的分组过滤条件有个瑕疵：第三组
`if 'transformer' not in n and 'bert' not in n` 没有排除 textcnn，
导致 CNN 分支的参数同时出现在第 1 组和第 3 组，学习率行为不确定。
本项目改成显式互斥的三个条件（`src/train/trainer.py::build_optimizer`）。

### 5. 为什么用加权采样器

论文已经对训练集做了负样本下采样（负样本 50400 → 4342），
但下采样后仍是正 2358 : 负 4342（约 1:1.8）。
`WeightedRandomSampler` 再平衡一层，让每个 batch 里两类数量接近 1:1，
减少模型偏向多数类（预测"无耀斑"）的倾向。
