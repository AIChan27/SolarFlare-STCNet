# ============================================================
# SupConLoss：有监督对比损失
# ============================================================

# from __future__ import annotations
#   作用：让类型注解（如 list[dict]）在 Python 3.9 及以下也能用
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 swin.py 导入全部内容（含 nn / F 等），保持与官方实现一致的写法
from src.models.swin import *          # noqa: F401,F403

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss（Khosla et al., 2020）。

    作用：让同一类别的样本在特征空间里互相靠近，不同类别的样本互相远离。
         对太阳耀斑这种正负极不平衡的任务，能让少数类（耀斑）的特征更紧凑，
         从而提升 TSS。

    Parameters
    ----------
    temperature : float
        温度系数，越小越"尖锐"。默认 0.1。
    scale_by_temperature : bool
        是否把 loss 乘回温度系数。默认 True。

    输入
    ----
    features : (B, D) 特征向量（本模型传入的是图像模态特征）
    labels   : (B,)   标签

    输出
    ----
    loss : 标量
    """

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        # L2 归一化，使点积等价于余弦相似度
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        # ---------- 构造 mask ----------
        # labels 和 mask 不能同时给
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # 无监督：只有自己和自己算同类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 有监督：label 相同则为同类，mask[i][j] = 1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # ---------- 计算两两相似度 ----------
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        # 数值稳定性：每行减去最大值再取指数
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # ---------- 正负样本掩码 ----------
        # logits_mask 去掉对角线（自己和自己）
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1.0 - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)

        # 分母 = 所有非自身的 exp 之和
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + \
                      torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        # 只在有正样本的行上求平均
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / \
                    num_positives_per_row[num_positives_per_row > 0]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        return loss.mean()