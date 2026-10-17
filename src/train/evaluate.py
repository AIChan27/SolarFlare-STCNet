"""
----------------------------------------------------------------------
src/train/evaluate.py
评估函数：计算太阳耀斑二分类的各项指标

指标说明（论文 4.2 节用的就是这几个）：
    ACC      : 总体准确率。类别极不平衡时会虚高（测试集正负比 1:80），
               所以只能参考，不能作为主要指标。
    TPR      : True Positive Rate = Recall = 报准率。
               真实耀斑中被正确预报出来的比例，耀斑预报第一优先。
    TNR      : True Negative Rate = Specificity = 非耀斑样本的报准率。
    FAR      : False Alarm Rate = 虚报率 = FPR。
               论文用"虚报率"这个词，定义与 FPR 相同。
    Precision: 预报为耀斑的样本里真的发生耀斑的比例。
    F1       : Precision 和 Recall 的调和平均。
    TSS      : True Skill Statistic = TPR - FPR。
               论文特别强调：TSS 对样本比例不平衡无偏，是本任务最有价值的指标。
    HSS      : Heidke Skill Score，另一个不平衡数据集的常用指标。
    AUC      : ROC 曲线下面积，阈值无关，综合衡量排序能力。

包含的函数：
    evaluate          ：在给定 DataLoader 上算全部指标
    print_metrics     ：终端友好地打印指标
    sweep_thresholds  ：扫描多个分类阈值，找 Recall/Precision 的平衡点
----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@torch.no_grad()
def evaluate(
    net,
    loader,
    device,
    threshold: float = 0.5,
    return_roc: bool = False,
):
    """
    在给定数据集上评估模型。

    Parameters
    ----------
    net : nn.Module
        模型（Mynet）。forward 返回 (图像特征, logits)，本函数只取 logits。
    loader : DataLoader
        验证集或测试集 DataLoader。
    device : torch.device
        CPU 或 CUDA。
    threshold : float
        分类阈值。概率 >= threshold 判为正类（有耀斑）。
        太阳耀斑预报里建议试 0.3 / 0.4 / 0.5，平衡报准率与虚报率。
    return_roc : bool
        是否额外返回 ROC 曲线数据 (fpr, tpr)，方便画图。

    Returns
    -------
    metrics : dict
        threshold / accuracy / precision / recall(tpr) / tnr / f1 /
        tss / hss / far / auc / confusion
        如果 return_roc=True，还包含 'roc': (fpr_list, tpr_list)。
        另外包含原始概率 'probs' 和标签 'labels'，方便外部画 PR 曲线或做阈值扫描。

    注意
    ----
    模型最后一层已经是 softmax，输出的是概率而不是 logits，
    所以这里不再做 softmax，直接取第 1 列（"有耀斑"的概率）。
    """
    net.eval()

    # 收集所有样本的正类概率和真实标签
    all_probs = []
    all_labels = []

    for batch in loader:
        # DataLoader 返回的是 ((img, params), label) 这种嵌套结构
        (imgs, params), labels = batch

        imgs = imgs.to(device)
        params = params.to(device)
        labels = labels.to(device)

        # forward 返回 (图像特征, 分类概率)
        _, probs = net((imgs, params))

        # probs[:, 1] 是"是耀斑"的概率
        all_probs.append(probs[:, 1].detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # 用阈值把概率转成 0/1 预测
    all_preds = (all_probs >= threshold).astype(int)

    # ---------- 基础指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)     # = TPR
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # ---------- 混淆矩阵 ----------
    # labels=[0,1] 保证即使某一类完全缺失，也返回 2×2 矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # ---------- TSS / HSS / FAR ----------
    # cm 结构：
    #   [[TN, FP],
    #    [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # = Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0      # = FAR
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # = Specificity
    tss = tpr - fpr

    # HSS 公式：2*(TP*TN - FP*FN) / [(TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)]
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    far = fpr   # 虚报率与 FPR 定义相同

    # ---------- AUC（阈值无关）----------
    # 需要正负样本都存在才能算
    if len(np.unique(all_labels)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),       # = TPR
        "tpr": float(tpr),
        "tnr": float(tnr),
        "f1": float(f1),
        "tss": float(tss),
        "hss": float(hss),
        "far": float(far),
        "auc": float(auc),
        "confusion": cm.tolist(),
        # 原始数据，方便外部复用（画 PR 曲线、扫描阈值等）
        "probs": all_probs,
        "labels": all_labels,
    }

    # 可选：返回 ROC 曲线数据
    if return_roc:
        if len(np.unique(all_labels)) < 2:
            metrics["roc"] = (None, None)
        else:
            fpr_curve, tpr_curve, _ = roc_curve(all_labels, all_probs)
            metrics["roc"] = (fpr_curve.tolist(), tpr_curve.tolist())

    return metrics


def print_metrics(metrics, title: str = "评估结果"):
    """
    打印指标，方便在终端查看。

    Parameters
    ----------
    metrics : dict
        evaluate() 返回的字典。
    title : str
        打印标题。
    """
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)
    print(f"阈值      : {metrics['threshold']:.2f}")
    print(f"ACC       : {metrics['accuracy']:.4f}   （总体准确率，不平衡时虚高）")
    print(f"TPR(报准率): {metrics['tpr']:.4f}   ← 耀斑预报第一优先")
    print(f"TNR       : {metrics['tnr']:.4f}   （非耀斑样本报准率）")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"TSS       : {metrics['tss']:.4f}   ← 论文最看重的指标")
    print(f"HSS       : {metrics['hss']:.4f}")
    print(f"FAR(虚报率): {metrics['far']:.4f}")
    print(f"AUC       : {metrics['auc']:.4f}")

    cm = metrics["confusion"]
    print("混淆矩阵 (labels=[0,1]，行=真实，列=预测):")
    print(f"           预测负   预测正")
    print(f"  真实负   TN={cm[0][0]:<7d} FP={cm[0][1]:<7d}")
    print(f"  真实正   FN={cm[1][0]:<7d} TP={cm[1][1]:<7d}")
    print("=" * 62)


def sweep_thresholds(
    net,
    loader,
    device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    verbose: bool = True,
):
    """
    扫描多个分类阈值，返回每个阈值下的指标。

    用途：太阳耀斑预报中，降低阈值能提高报准率(TPR)但会增加虚报率(FAR)，
         扫描一遍可以找到适合业务需求的平衡点。

    Parameters
    ----------
    thresholds : tuple
        要扫描的阈值列表。

    Returns
    -------
    results : list[dict]
        每个阈值对应的指标字典（已剔除 probs/labels 大数组）。
    """
    # 只在第一次调用时跑一遍模型，后面复用概率，避免重复推理
    base = evaluate(net, loader, device, threshold=0.5)
    probs, labels = base["probs"], base["labels"]

    results = []
    if verbose:
        print()
        print("=" * 78)
        print("阈值扫描")
        print("=" * 78)
        print(f"{'阈值':>6} | {'TPR':>7} {'TNR':>7} {'Prec':>7} {'F1':>7} "
              f"{'TSS':>7} {'FAR':>7} {'ACC':>7}")
        print("-" * 78)

    for th in thresholds:
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        m = {
            "threshold": float(th),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "tss": float(tpr - fpr),
            "far": float(fpr),
            "accuracy": float(accuracy_score(labels, preds)),
        }
        results.append(m)

        if verbose:
            print(f"{th:>6.2f} | {m['tpr']:>7.4f} {m['tnr']:>7.4f} "
                  f"{m['precision']:>7.4f} {m['f1']:>7.4f} "
                  f"{m['tss']:>7.4f} {m['far']:>7.4f} {m['accuracy']:>7.4f}")

    if verbose:
        # 顺手报一下 TSS 最高的阈值
        best = max(results, key=lambda x: x["tss"])
        print("-" * 78)
        print(f"TSS 最高: 阈值={best['threshold']:.2f}, "
              f"TSS={best['tss']:.4f}, TPR={best['tpr']:.4f}, FAR={best['far']:.4f}")
        print("=" * 78)

    return results


def select_threshold(
    net,
    loader,
    device,
    metric: str = "tss",
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    verbose: bool = True,
):
    """
    在【验证集】上扫描分类阈值并选出最优值。

    ⚠️ 为什么必须用验证集，不能用测试集
    ----------------------------------
    阈值是一个"被选出来的超参数"。如果直接在测试集上扫阈值再挑最好的，
    等价于用测试集的答案去挑一个对这批测试样本最有利的分割点 ——
    这是典型的**测试集泄漏（test leakage）**，会让报告指标虚高。

    实测过它的危害：在 SOLAR-STORM2 上，TSS 在阈值 0.4~0.7 之间摆动了 0.04，
    而"虚高 0.04"比我们想验证的任何模型改进（0.01~0.02）都大。
    也就是说，不修这个问题，后续所有"改进"结论都可能被污染。

    正确流程：
        1. 验证集上扫阈值 → 选出 t*
        2. t* 固定下来
        3. 用 t* 在测试集上**只评估一次** → 这个数字才能报告

    Parameters
    ----------
    net : nn.Module
    loader : DataLoader
        【必须是验证集】。传测试集就是泄漏。
    device : torch.device
    metric : str
        选阈值的判据，'tss'（默认，论文主指标）/ 'f1' / 'precision'。
        注意：不建议用 'far' 或 'accuracy' 作为判据。
    thresholds : tuple
        候选阈值。
    verbose : bool

    Returns
    -------
    best_threshold : float
        选出的最优阈值。
    best_row : dict
        该阈值下的指标（在验证集上）。
    all_rows : list[dict]
        全部阈值的扫描结果（便于画图/存盘）。
    """
    # 只推理一次，拿到全部概率后在各阈值上复用
    results = sweep_thresholds(net, loader, device, thresholds=thresholds, verbose=False)

    key = metric.lower()
    if key not in ("tss", "f1", "precision", "accuracy", "tnr", "tpr"):
        raise ValueError(
            f"不支持的选阈值判据 {metric!r}；可选 tss / f1 / precision / accuracy / tnr / tpr"
        )

    best_row = max(results, key=lambda r: r[key])
    best_threshold = float(best_row["threshold"])

    if verbose:
        print()
        print("=" * 78)
        print(f"验证集阈值选择（判据 = {key.upper()}）")
        print("=" * 78)
        print(f"{'阈值':>6} | {'TPR':>7} {'TNR':>7} {'Prec':>7} {'F1':>7} "
              f"{'TSS':>7} {'FAR':>7} {'ACC':>7}")
        print("-" * 78)
        for r in results:
            mark = "  ← 选中" if abs(r["threshold"] - best_threshold) < 1e-9 else ""
            print(f"{r['threshold']:>6.2f} | {r['tpr']:>7.4f} {r['tnr']:>7.4f} "
                  f"{r['precision']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['tss']:>7.4f} {r['far']:>7.4f} {r['accuracy']:>7.4f}{mark}")
        print("-" * 78)
        print(f"选中阈值 t* = {best_threshold:.2f}（验证集 {key.upper()} = {best_row[key]:.4f}）")
        print("该阈值将固定用于测试集评估，测试集不再参与任何阈值搜索。")
        print("=" * 78)

    return best_threshold, best_row, results


# ============================================================
# 简单自测
# ============================================================

if __name__ == "__main__":
    # 造一个假模型和假数据，检查指标计算是否正常
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DummyNet(nn.Module):
        """假模型：直接输出随机概率，只为验证评估流程能跑通。"""

        def forward(self, inx):
            imgs = inx[0]
            probs = torch.rand(imgs.size(0), 2)
            probs = probs / probs.sum(dim=1, keepdim=True)
            return imgs.new_zeros(imgs.size(0), 320), probs

    # 构造 200 条假样本
    imgs = torch.randn(200, 1, 224, 224)
    params = torch.randn(200, 1, 10)
    labels = torch.randint(0, 2, (200,))

    ds = TensorDataset(imgs, params, labels)

    # 包一层，让 batch 结构变成 ((img, params), label)，与真实 DataLoader 一致
    class Wrap:
        def __init__(self, ds):
            self.ds = ds

        def __iter__(self):
            for a, b, c in self.ds:
                yield (a.unsqueeze(0), b.unsqueeze(0)), c.unsqueeze(0)

        def __len__(self):
            return len(self.ds)

    m = evaluate(DummyNet(), Wrap(ds), torch.device("cpu"))
    print_metrics(m, "自测（随机模型，指标应接近随机水平）")
