"""
----------------------------------------------------------------------
src/models/stcnet.py
STCNet：Swin Transformer + 1D-CNN 的多模态太阳耀斑预报模型

对应论文：
    李蓉等《基于多模态特征融合的太阳耀斑预报模型》
    中国科学：物理学 力学 天文学, 2024, 54(12): 129611

模型结构（论文 3.1 节，三个模块）：
    第一部分  原始图像特征提取网络：Swin Transformer
              输入 太阳光球纵向磁图 (1, 224, 224)
              输出 一维磁图特征向量 (B, 320)
    第二部分  磁场特征参数提取网络：1D-CNN
              输入 10 个 SHARP 磁场参数 (B, 1, 10)
              输出 一维参数特征向量 (B, 320)
    第三部分  分类网络：拼接两个模态的特征 → 全连接层 → 2 分类
              输入 (B, 320+320=640) → 输出 (B, 2)

额外：SupConLoss（有监督对比学习）作为辅助损失，
      让同类别样本在特征空间里聚拢，提升小样本正类的可分性。

本文件相比原代码 models.py 的修改（都是修 bug，模型结构未动）：
    1. Mynet.__init__ 按 config.swin_variant 选择主干，不再硬编码 base；
       （原始代码硬编码 swin_base，但实际训练出的 model-27/29.pth 里
         patch_embed 形状是 (96,1,4,4)，证明真正用的是 Tiny）
    2. 预训练权重的加载从 forward() 移到 __init__。
       原代码在 forward 里每个 batch 都 torch.load 一次 336 MB 的权重文件，
       即使路径修对了，训练也会慢到不可用。
    3. 加载权重时取 sd["model"]（原代码漏了这一步，导致权重一个都没装上）。
    4. 保留返回 (图像特征, logits) 的接口，因为 SupConLoss 要用图像特征。
----------------------------------------------------------------------
"""

# from __future__ import annotations
#   作用：让类型注解（如 list[dict]）在 Python 3.9 及以下也能用
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 swin.py 导入全部内容（含 nn / F 等），保持与官方实现一致的写法
from src.models.swin import *          # noqa: F401,F403

from src.models.cnn import CNN


# ============================================================
# 完整多模态模型
# ============================================================


class Mynet(nn.Module):
    """
    STCNet 完整模型（多模态融合）。

    Parameters
    ----------
    config : Config
        全局配置对象，用到以下字段：
            swin_variant      : Swin 型号名
            swin_feature_dim  : Swin 输出维度（默认 320）
            cnn_feature_dim   : CNN 分支输出维度（默认 320）
            num_classes       : 类别数（2）
            dropout           : dropout 比例
    pretrained_path : str | Path | None
        Swin 预训练权重路径。None 表示不加载（从头训练，用于消融实验）。

    前向传播
    --------
    输入  inx = (img, params)
              img    : (B, 1, 224, 224)
              params : (B, 1, 10)
    输出  (fea, logits)
              fea    : (B, 320) 图像模态特征（给 SupConLoss 用）
              logits : (B, 2)   分类概率（已过 softmax）
    """

    def __init__(self, config, pretrained_path=None):
        super(Mynet, self).__init__()
        self.config = config

        # ---------- 第一部分：Swin Transformer 主干 ----------
        # 按 config.swin_variant 选择型号，不再硬编码
        #   'swin_tiny_patch4_window7_224' → embed_dim 96，主干 2756 万参数
        #   'swin_base_patch4_window7_224' → embed_dim 128，主干 8810 万参数
        # 注意：swin.py 里的工厂函数默认 in_chans=1，就是为了吃单通道灰度磁图
        variant = getattr(config, "swin_variant", "swin_tiny_patch4_window7_224")
        if variant not in globals():
            raise ValueError(
                f"未知的 Swin 型号: {variant}；可选: "
                f"swin_tiny_patch4_window7_224 / swin_base_patch4_window7_224 等"
            )
        swin_factory = globals()[variant]
        self.transformer = swin_factory(getattr(config, "swin_feature_dim", 320))

        # 加载预训练权重（关键修复点，详见 load_pretrained 的说明）
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        # ---------- 第二部分：磁场参数 1D-CNN 分支 ----------
        # 特征个数从 config.num_params 取，不能写死：
        #   SOLAR-STORM2 → 10 个 SHARP 参数
        #   JW-FD        → 29 个磁场特征（梯度7 + 中性线13 + 小波5 + 磁通量4）
        # 传入后会决定 CNN.fc 的输入维度（32*num_params）
        num_params = getattr(config, "num_params", 10)
        cnn_out_dim = getattr(config, "cnn_feature_dim", 320)
        self.text_cnn = CNN(num_params=num_params, out_dim=cnn_out_dim)

        # ---------- 第三部分：分类网络 ----------
        # 两个模态特征拼接后长度 = swin_feature_dim + cnn_feature_dim
        feat_dim = (
            getattr(config, "swin_feature_dim", 320)
            + getattr(config, "cnn_feature_dim", 320)
        )
        self.fc_1 = nn.Linear(feat_dim, config.num_classes)
        self.drop = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=1)

    # --------------------------------------------------------
    # 预训练权重加载
    # --------------------------------------------------------

    def load_pretrained(self, path, verbose: bool = True):
        """
        加载 Swin 预训练权重。

        这是最关键的一个修复点。

        为什么原代码加载不上
        --------------------
        预训练权重文件内部的顶层结构是：
            {'model': OrderedDict([('patch_embed.proj.weight', ...), ...])}
        也就是真正的 state_dict 藏在外层的 'model' 键里面。

        原代码写的是：
            pretrained_weights = torch.load('swin_base_patch4_window7_224.pth')
            self.transformer.load_state_dict(pretrained_weights, False)

        它把整个 {'model': ...} 字典当成 state_dict 传进去，
        又用了 strict=False，于是 PyTorch 发现"键名全都不匹配"也不报错，
        静默地什么都不加载 —— 表现就是训练能跑，但预训练完全没生效
        （实测 patch_embed.proj.weight 装载前后完全没变）。

        正确写法是取 ["model"]，并剔除分类头（head/fc）的键，
        因为我们要用自己的 320 维输出，而不是 ImageNet 的 1000 类。

        Parameters
        ----------
        path : str | Path
            权重文件路径。
        verbose : bool
            是否打印加载统计。

        Returns
        -------
        msg : _IncompatibleKeys
            load_state_dict 的返回值，含 missing_keys / unexpected_keys。
        """
        # weights_only=False：该权重是普通 state_dict 包了一层 dict，
        # 在新版 PyTorch（>=2.6 默认 weights_only=True）下需要显式放开
        state = torch.load(path, map_location="cpu", weights_only=False)

        # 兼容两种结构：{'model': sd} 和 直接就是 sd
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            weights_dict = state["model"]
            if verbose:
                print(
                    f"[权重] 已从 {str(path).split(chr(92))[-1]} 取出内层 state_dict"
                    f"（{len(weights_dict)} 个条目）"
                )
        else:
            weights_dict = state
            if verbose:
                print(f"[权重] {str(path).split(chr(92))[-1]} 直接就是 state_dict（{len(weights_dict)} 个条目）")

        # 剔除分类头：Swin 官方实现里叫 head，本文件用的 swin.py 里叫 fc
        for k in list(weights_dict.keys()):
            if k.startswith("head.") or k.startswith("fc."):
                del weights_dict[k]

        msg = self.transformer.load_state_dict(weights_dict, strict=False)

        if verbose:
            n_model = len(self.transformer.state_dict())
            n_loaded = n_model - len(msg.missing_keys)
            print(f"[权重] 主干参数 {n_loaded}/{n_model} 个已加载")
            if msg.missing_keys:
                print(f"[权重] 未加载的键（正常，通常是分类头）: {msg.missing_keys}")
            if msg.unexpected_keys:
                # Swin 官方实现会注册 attn_mask 缓冲区，本实现没有，属正常
                print(
                    f"[权重] 权重里多余的键 {len(msg.unexpected_keys)} 个"
                    f"（通常为 attn_mask 缓冲，可忽略）"
                )
        return msg

    # --------------------------------------------------------
    # 前向传播
    # --------------------------------------------------------

    def forward(self, inx):
        """
        Parameters
        ----------
        inx : tuple
            (img, params)
            img    : (B, 1, 224, 224) 磁图
            params : (B, 1, 10)       磁场参数

        Returns
        -------
        (fea, logits) : tuple[Tensor, Tensor]
            fea    : (B, swin_feature_dim) 图像模态特征，供对比学习使用
            logits : (B, num_classes)      分类概率
        """
        img = inx[0]        # 第一个元素是图像数据
        tokens = inx[1]     # 第二个元素是磁场参数

        # ---------- 1. 图像模态 ----------
        # 注意：预训练权重已经在 __init__ 里加载完了，
        #       这里绝不能再 torch.load（原代码每 forward 一次就 load 一遍 336 MB）
        img = self.transformer(img)             # (B, 320)

        # ---------- 2. 参数模态 ----------
        tokens = tokens.float()                 # (B, 1, 10)
        text_features = self.text_cnn(tokens)   # (B, 320)
        text_features = self.drop(text_features)

        # ---------- 3. 跨模态拼接 + 分类 ----------
        fea = torch.cat([img, text_features], dim=1)   # (B, 640)
        logits = self.fc_1(fea)                        # (B, 2)
        logits = self.softmax(logits)                  # 转成概率

        # 返回图像特征和分类概率（图像特征给 SupConLoss 用）
        return img, logits

# ============================================================
# 简单自测
# ============================================================

if __name__ == "__main__":
    from config import Config, SWIN_PRETRAINED

    cfg = Config()
    print("=" * 60)
    print("Mynet 构造与形状自测")
    print("=" * 60)

    model = Mynet(cfg, pretrained_path=SWIN_PRETRAINED)
    model.eval()

    # 造一个假 batch
    img = torch.randn(2, 1, 224, 224)
    params = torch.randn(2, 1, 10)

    with torch.no_grad():
        fea, logits = model((img, params))

    print()
    print(f"输入图像  : {tuple(img.shape)}")
    print(f"输入参数  : {tuple(params.shape)}")
    print(f"图像特征  : {tuple(fea.shape)}")
    print(f"分类概率  : {tuple(logits.shape)}")
    print(f"概率和    : {logits.sum(dim=1).tolist()}（应接近 1.0）")
    print(f"总参数量  : {sum(p.numel() for p in model.parameters()):,}")
