"""
----------------------------------------------------------------------
src/utils/visualization.py
可视化工具：训练曲线、ROC 曲线、混淆矩阵、预测样例

产出图统一保存到 experiments/figures/，文件名带时间戳前缀便于对比不同实验。

包含的函数：
    plot_training_curves  ：loss / acc / tss / far 随 epoch 变化
    plot_confusion_matrix ：混淆矩阵热力图
    plot_roc_curve        ：ROC 曲线
    plot_prediction_samples：抽样展示预测结果（对/错标注在标题上）
    save_metrics_json     ：把指标字典（剔除大数组）存成 JSON
----------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

# 无 GUI 环境下也能出图（服务器/批处理）
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------- 中文字体设置 ----------
# matplotlib 默认字体 DejaVu Sans 没有中文字形，图标题里的中文会变成
# 一堆 "Glyph xxx missing from font(s)" 警告，且图上显示为空白方块。
# 这里按优先级挑一个系统里存在的中文字体；都没有就退回默认并关闭警告。
from matplotlib import font_manager

_CN_FONT_CANDIDATES = [
    "Microsoft YaHei",      # 微软雅黑（Windows 10/11 自带）
    "SimHei",               # 黑体（Windows 自带）
    "Noto Sans CJK SC",     # Linux 常见
    "Source Han Sans SC",   # 思源黑体
    "PingFang SC",          # macOS
]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _f in _CN_FONT_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
else:
    # 一个都没有：关掉缺字警告，避免刷屏（图上中文仍会缺失）
    import warnings

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

# 让负号能正常显示（使用中文字体后必须设置，否则负号会变成方块）
plt.rcParams["axes.unicode_minus"] = False

from config import FIGURES_DIR, REPORTS_DIR
from src.utils.paths import ensure_dir


def _timestamp() -> str:
    """返回形如 20260101_120000 的时间戳，用于给输出文件命名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fig_path(name: str, out_dir: Path | None = None) -> Path:
    """
    生成输出图片路径，父目录不存在会自动创建。

    Parameters
    ----------
    name : str
        文件名（不含目录）。
    out_dir : Path | None
        输出目录，默认 experiments/figures/。

    Returns
    -------
    Path
    """
    out_dir = ensure_dir(out_dir or FIGURES_DIR)
    return out_dir / name


# ============================================================
# 1. 训练曲线
# ============================================================


def plot_training_curves(history: dict, tag: str = "stcnet", out_dir: Path | None = None):
    """
    画训练曲线：loss / acc / tss / far 各一格。

    Parameters
    ----------
    history : dict
        trainer.train() 返回的 history，键含 train_loss / train_acc /
        val_acc / val_f1 / val_tss / val_far / lr。
    tag : str
        实验标签，用于文件名和图标题。
    out_dir : Path | None

    Returns
    -------
    save_path : Path
    """
    # 两个子图：左 loss，右 指标
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ---------- 左上：loss ----------
    ax = axes[0][0]
    ax.plot(epochs, history["train_loss"], "o-", ms=3, label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # ---------- 右上：accuracy ----------
    ax = axes[0][1]
    ax.plot(epochs, history["train_acc"], "o-", ms=3, label="train acc")
    if history.get("val_acc"):
        ax.plot(epochs, history["val_acc"], "s-", ms=3, label="val acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    # ---------- 左下：TSS / F1 ----------
    ax = axes[1][0]
    if history.get("val_tss"):
        ax.plot(epochs, history["val_tss"], "^-", ms=3, label="val TSS")
    if history.get("val_f1"):
        ax.plot(epochs, history["val_f1"], "v-", ms=3, label="val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation TSS / F1（TSS 是本任务最看重的指标）")
    ax.grid(alpha=0.3)
    ax.legend()

    # ---------- 右下：FAR 与学习率 ----------
    ax = axes[1][1]
    if history.get("val_far"):
        ax.plot(epochs, history["val_far"], "d-", ms=3, color="tab:red", label="val FAR")
        ax.set_ylabel("FAR", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["lr"], "--", color="tab:gray", label="lr")
    ax2.set_ylabel("Learning rate", color="tab:gray")
    ax2.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title("False Alarm Rate（虚报率）与学习率")
    ax.grid(alpha=0.3)

    fig.suptitle(f"STCNet 训练曲线 - {tag}", fontsize=13)
    fig.tight_layout()

    save_path = _fig_path(f"{_timestamp()}_{tag}_curves.png", out_dir)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"[图] 训练曲线已保存: {save_path}")
    return save_path


# ============================================================
# 2. 混淆矩阵
# ============================================================


def plot_confusion_matrix(cm, tag: str = "stcnet", out_dir: Path | None = None):
    """
    画混淆矩阵热力图。

    Parameters
    ----------
    cm : list | np.ndarray
        2×2 混淆矩阵，行=真实，列=预测。[[TN, FP], [FN, TP]]
    tag : str
    out_dir : Path | None

    Returns
    -------
    save_path : Path
    """
    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测: 无耀斑", "预测: 有耀斑"])
    ax.set_yticklabels(["真实: 无耀斑", "真实: 有耀斑"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {tag}")

    # 在每格写数值
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    save_path = _fig_path(f"{_timestamp()}_{tag}_confusion.png", out_dir)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"[图] 混淆矩阵已保存: {save_path}")
    return save_path


# ============================================================
# 3. ROC 曲线
# ============================================================


def plot_roc_curve(probs, labels, auc_value: float, tag: str = "stcnet", out_dir: Path | None = None):
    """
    画 ROC 曲线。

    Parameters
    ----------
    probs : np.ndarray
        正类预测概率。
    labels : np.ndarray
        真实标签。
    auc_value : float
        AUC 值，打印在图例里。
    tag : str
    out_dir : Path | None

    Returns
    -------
    save_path : Path | None
        某一类样本缺失时返回 None（无法画 ROC）。
    """
    if len(np.unique(labels)) < 2:
        print("[图] 样本只有单一类别，跳过 ROC 曲线")
        return None

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"ROC (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (= FAR 虚报率)")
    ax.set_ylabel("True Positive Rate (报准率)")
    ax.set_title(f"ROC Curve - {tag}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_path = _fig_path(f"{_timestamp()}_{tag}_roc.png", out_dir)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"[图] ROC 曲线已保存: {save_path}")
    return save_path


# ============================================================
# 4. 预测样例
# ============================================================


def plot_prediction_samples(
    net,
    loader,
    device,
    tag: str = "stcnet",
    n: int = 8,
    out_dir: Path | None = None,
    random_pick: bool = True,
    seed: int = 0,
    show_errors: bool = False,
):
    """
    抽样展示模型的预测结果，标题标明 真实标签/预测标签/概率，错例用红色标题。

    用途：直观检查模型是不是学到了磁图的形态特征，还是只靠磁场参数。

    Parameters
    ----------
    net : nn.Module
    loader : DataLoader
        建议传 test 集 DataLoader。
    device : torch.device
    n : int
        展示多少张。
    tag : str
    out_dir : Path | None
    random_pick : bool
        True  = 从整个数据集里随机抽 n 张（默认，更有代表性）
        False = 取前 n 张（数据按时间排序，会全是同一时期的样本）
    seed : int
        随机抽样种子，保证图可复现。
    show_errors : bool
        True = 优先展示预测错误的样本（分析错例用）。

    Returns
    -------
    save_path : Path | None
    """
    import random as _random

    import torch

    net.eval()

    collected = []      # [(img, true_label, pred_label, prob), ...]
    all_probs, all_labels, all_imgs = [], [], []

    # 先把整个数据集跑一遍，收集概率和标签（为了能随机抽/挑错例）
    with torch.no_grad():
        for batch in loader:
            (imgs, params), labels = batch
            _, probs = net((imgs.to(device), params.to(device)))
            probs = probs[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_imgs.append(imgs[:, 0].clone())      # 只留单通道，用于画图

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_imgs = torch.cat(all_imgs, dim=0)            # (N, H, W)
    all_preds = (all_probs >= 0.5).astype(int)

    # 选哪些下标来展示
    if show_errors:
        # 优先展示错例
        err_idx = np.where(all_preds != all_labels)[0]
        if len(err_idx) >= n:
            pick = np.random.default_rng(seed).choice(err_idx, size=n, replace=False)
        else:
            rest = np.setdiff1d(np.arange(len(all_labels)), err_idx)
            extra = np.random.default_rng(seed).choice(rest, size=n - len(err_idx), replace=False)
            pick = np.concatenate([err_idx, extra])
    elif random_pick:
        pick = np.random.default_rng(seed).choice(
            len(all_labels), size=min(n, len(all_labels)), replace=False
        )
    else:
        pick = np.arange(min(n, len(all_labels)))

    for i in pick:
        collected.append(
            (all_imgs[i].numpy(), int(all_labels[i]), int(all_preds[i]), float(all_probs[i]))
        )

    if not collected:
        print("[图] 没有可用样本，跳过预测样例")
        return None

    cols = 4
    rows = int(np.ceil(len(collected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, (img, y_true, y_pred, prob) in zip(axes, collected):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        correct = y_true == y_pred
        # 标签含义：1 = 48h 内有 M 级及以上耀斑
        title = f"真={y_true} 预={y_pred} p={prob:.3f}"
        ax.set_title(title, fontsize=9, color="green" if correct else "red")

    # 多余的子图关掉
    for ax in axes[len(collected):]:
        ax.axis("off")

    note = "错例优先" if show_errors else ("随机抽样" if random_pick else "前 N 条")
    fig.suptitle(f"预测样例 - {tag}（{note}；绿=预测正确，红=预测错误）", fontsize=12)
    fig.tight_layout()

    save_path = _fig_path(f"{_timestamp()}_{tag}_samples.png", out_dir)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"[图] 预测样例已保存: {save_path}")
    return save_path


# ============================================================
# 5. 指标落盘
# ============================================================


def save_metrics_json(metrics: dict, tag: str = "stcnet", out_dir: Path | None = None):
    """
    把指标字典存成 JSON（自动剔除 probs/labels 这类大数组）。

    Parameters
    ----------
    metrics : dict
        evaluate() 的返回字典。
    tag : str
    out_dir : Path | None
        默认 experiments/reports/。

    Returns
    -------
    save_path : Path
    """
    out_dir = ensure_dir(out_dir or REPORTS_DIR)

    # 只保留可序列化的标量/列表，剔除 numpy 大数组
    clean = {}
    for k, v in metrics.items():
        if k in ("probs", "labels"):
            continue
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v

    save_path = out_dir / f"{_timestamp()}_{tag}_metrics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print(f"[记录] 指标已保存: {save_path}")
    return save_path


# ============================================================
# 简单自测
# ============================================================

if __name__ == "__main__":
    # 造一份假的 history 和混淆矩阵，验证画图流程
    fake_history = {
        "train_loss": [0.7, 0.6, 0.5, 0.45, 0.4],
        "train_acc": [0.6, 0.7, 0.75, 0.8, 0.85],
        "val_acc": [0.65, 0.72, 0.78, 0.82, 0.86],
        "val_f1": [0.3, 0.45, 0.55, 0.6, 0.65],
        "val_tss": [0.2, 0.4, 0.5, 0.55, 0.6],
        "val_far": [0.4, 0.3, 0.2, 0.15, 0.1],
        "lr": [2e-4, 2e-4, 1e-4, 1e-4, 5e-5],
    }
    plot_training_curves(fake_history, tag="selftest")

    fake_cm = [[11500, 352], [50, 146]]
    plot_confusion_matrix(fake_cm, tag="selftest")

    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0, 0.4, 900), rng.uniform(0.5, 1.0, 100)])
    labels = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    plot_roc_curve(probs, labels, auc_value=0.91, tag="selftest")

    save_metrics_json({"accuracy": 0.97, "tss": 0.75, "confusion": fake_cm}, tag="selftest")
    print("\n自测完成，请查看 experiments/figures/ 与 experiments/reports/")
