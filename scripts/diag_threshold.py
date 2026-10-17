"""
======================================================================
scripts/diag_threshold.py
诊断工具：画出"同一份权重"在验证集与测试集上的 TSS-阈值曲线
======================================================================

⚠️ 这个脚本的用途是【理解曲线形状】，不是用来挑阈值。

    挑阈值必须只用验证集（见 `python main.py test --val-threshold`）。
    本脚本把测试集的 TSS 也打出来，是为了让你看清：
        - 验证集选出的最优点，在测试集上到底好不好
        - 曲线在这一段是"尖峰"还是"平台"
    如果曲线是平台，说明阈值在这一区间怎么选都差不多，
    那 val 选出来的 0.01 和 0.02 的差异基本是噪声 —— 这本身就是有用的信息。

为什么需要这个工具
------------------
2026-10-17 实测遇到一个典型现象：
    val 在 0.01 处 TSS 最高（0.6707），但测试集在 0.01 反而比 0.02 略差。
    这说明低阈值区间是一个平台，val 的最优点带有噪声。
    只看最终一个数字是无法发现这件事的，必须看整条曲线。

用法
----
    # 用默认阈值网格
    python scripts/diag_threshold.py --ckpt experiments/checkpoints/jwfd_nows_best.pth

    # 指定 tag（自动找 <tag>_best.pth）
    python scripts/diag_threshold.py --tag jwfd_nows

    # 自定义阈值网格
    python scripts/diag_threshold.py --tag jwfd_nows --thresholds 0.005 0.01 0.02 0.05 0.1 0.2 0.5
======================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

import config
from config import Config
from src.data.dataset import SolarFlareDataset, build_transform, compute_param_range
from src.models.stcnet import Mynet
from src.train.evaluate import evaluate


def build_loader(csv_path, kind: str, cfg) -> tuple[DataLoader, int]:
    pmin, pmax = compute_param_range(config.TRAIN_CSV)
    ds = SolarFlareDataset(
        csv_path, img_size=cfg.img_size, param_min=pmin, param_max=pmax,
        transform=build_transform(kind, cfg.img_size),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)
    return loader, len(ds)


def conf_at(probs, labels, th: float):
    """给定阈值算 (TPR, TSS, FAR, Precision)。"""
    pred = (probs >= th).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return tpr, tpr - fpr, fpr, prec


def main():
    ap = argparse.ArgumentParser(
        description="诊断：验证集 vs 测试集的 TSS-阈值曲线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ckpt", type=str, default=None, help="权重路径")
    ap.add_argument("--tag", type=str, default="jwfd_nows", help="实验标签（自动找 <tag>_best.pth）")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="自定义阈值网格（默认用 config.THRESHOLD_SWEEP）")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else config.CHECKPOINTS_DIR / f"{args.tag}_best.pth"
    if not ckpt.exists():
        print(f"[错误] 找不到权重: {ckpt}")
        return 1

    thresholds = args.thresholds if args.thresholds else list(config.THRESHOLD_SWEEP)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print(f"  阈值曲线诊断 | 权重: {ckpt.name}")
    print("  ※ 测试集一列仅用于理解曲线形状，挑阈值必须只用验证集")
    print("=" * 88)

    # ---------- 模型 ----------
    model = Mynet(cfg, pretrained_path=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # ---------- 两个数据集各推理一次 ----------
    val_loader, n_val = build_loader(config.VAL_CSV, "val", cfg)
    test_loader, n_test = build_loader(config.TEST_CSV, "test", cfg)
    print(f"验证集 {n_val:,} 条 | 测试集 {n_test:,} 条\n")

    rv = evaluate(model, val_loader, device, threshold=0.5)
    rt = evaluate(model, test_loader, device, threshold=0.5)
    pv, lv = rv["probs"], rv["labels"]
    pt, lt = rt["probs"], rt["labels"]

    # ---------- 表格 ----------
    header = (f"{'阈值':>7} | {'val_TPR':>7} {'val_FAR':>7} {'val_TSS':>7} | "
              f"{'test_TPR':>8} {'test_FAR':>8} {'test_TSS':>8} | 备注")
    print(header)
    print("-" * len(header))

    # 先找 val 最优（判据只用验证集）
    val_rows = [(th, *conf_at(pv, lv, th)) for th in thresholds]
    best_th = max(val_rows, key=lambda r: r[2])[0]

    for th in thresholds:
        v_tpr, v_tss, v_far, _ = conf_at(pv, lv, th)
        t_tpr, t_tss, t_far, _ = conf_at(pt, lt, th)
        marks = []
        if th == best_th:
            marks.append("← val 最优")
        print(f"{th:>7.3f} | {v_tpr:>7.4f} {v_far:>7.4f} {v_tss:>7.4f} | "
              f"{t_tpr:>8.4f} {t_far:>8.4f} {t_tss:>8.4f} | {'  '.join(marks)}")

    print("-" * len(header))

    # ---------- 曲线形状判断 ----------
    top = sorted(val_rows, key=lambda r: r[2], reverse=True)
    best_v_tss = top[0][2]
    plateau = [r[0] for r in top if best_v_tss - r[2] <= 0.01]
    plateau.sort()

    print(f"验证集最优阈值 t* = {best_th:.3f}（val TSS = {best_v_tss:.4f}）")
    print(f"距最优 TSS 相差 <0.01 的阈值区间: {plateau[0]:.3f} ~ {plateau[-1]:.3f}"
          f"（共 {len(plateau)} 个点）")

    if len(plateau) >= 3:
        print("→ 结论：这是一个【平台】。阈值在这一段怎么选都差不多，val 的最优点带噪声。")
    else:
        print("→ 结论：这是一个【尖峰】。最优点比较明确，val 选阈值可信度较高。")

    # 边界警告
    if abs(best_th - thresholds[0]) < 1e-9:
        print(f"⚠️ 警告：最优点落在扫描区间【下边界】{thresholds[0]}，"
              f"真正最优点可能更低，建议把 --thresholds 往下扩。")
    if abs(best_th - thresholds[-1]) < 1e-9:
        print(f"⚠️ 警告：最优点落在扫描区间【上边界】{thresholds[-1]}，"
              f"真正最优点可能更高，建议把 --thresholds 往上扩。")

    print(f"\n验证集 AUC = {rv['auc']:.4f} | 测试集 AUC = {rt['auc']:.4f}")
    print("（AUC 是阈值无关的，可以直接作为「模型判别能力」的指标来比较不同实验）")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
