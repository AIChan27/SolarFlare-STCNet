"""
----------------------------------------------------------------------
main.py
项目主入口，按阶段调用 pipeline

用法（在项目根目录运行）：
    python main.py check                     # 检查环境与数据是否就绪
    python main.py build-data                # 阶段一：构建索引 CSV
    python main.py build-data --no-downsample  # 构建但不对负样本下采样
    python main.py train                     # 阶段二：训练
    python main.py train --smoke             # 冒烟测试（小样本 1 个 epoch）
    python main.py test --ckpt <权重路径>     # 阶段三：测试集评估
    python main.py test --sweep              # 评估 + 阈值扫描
    python main.py all                       # 一条龙：build-data → train → test

常用可选参数：
    --epochs N      覆盖 epoch 数
    --batch-size N  覆盖 batch 大小
    --no-supcon     关闭对比学习（消融实验）
    --tag NAME      实验标签，用于给输出文件命名
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证 `python main.py` 在任意位置运行都能 import 到包
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import config
from config import Config


# ============================================================
# 阶段零：环境与数据自检
# ============================================================


def cmd_check(args):
    """
    检查环境、数据、权重是否就绪，并把关键信息打印出来。

    建议第一次跑项目先执行这个命令，能提前发现路径问题和环境问题。
    """
    print("=" * 70)
    print("环境与数据自检")
    print("=" * 70)

    # ---------- 1. Python 与依赖 ----------
    print("\n[1] 依赖版本")
    print(f"    Python      : {sys.version.split()[0]}")
    print(f"    PyTorch     : {torch.__version__}")
    print(f"    NumPy       : {np.__version__}")
    print(f"    CUDA 可用   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU         : {torch.cuda.get_device_name(0)}")
    try:
        import torchvision
        print(f"    torchvision : {torchvision.__version__}")
    except ImportError:
        print("    torchvision : 未安装（必须装，Dataset 用到 transforms）")
    for mod in ("sklearn", "PIL", "pandas", "matplotlib"):
        try:
            __import__(mod)
            print(f"    {mod:12s}: 已安装")
        except ImportError:
            print(f"    {mod:12s}: 缺失（请安装）")

    # ---------- 2. 数据文件 ----------
    print("\n[2] 数据文件")
    paths = {
        "磁图目录": config.MAGNETOGRAM_DIR,
        "参数表": config.PARA_FLARE_TXT,
        "Swin 预训练权重": config.SWIN_PRETRAINED,
        "CNN 预训练权重": config.CNN_PRETRAINED,
    }
    all_ok = True
    for name, p in paths.items():
        ok = Path(p).exists()
        all_ok &= ok
        print(f"    [{'OK ' if ok else '缺失'}] {name:16s} {p}")

    # 磁图数量核对（官方全量应为 73810 张）
    if Path(config.MAGNETOGRAM_DIR).exists():
        n_jpg = sum(1 for _ in Path(config.MAGNETOGRAM_DIR).glob("*.jpg"))
        flag = "OK " if n_jpg == 73810 else "注意"
        print(f"    [{flag}] 磁图数量 {n_jpg} 张（官方全量为 73810 张）")

    # ---------- 3. 索引 CSV ----------
    print("\n[3] 索引 CSV（阶段一产物）")
    for name, p in [
        ("训练集", config.TRAIN_CSV),
        ("验证集", config.VAL_CSV),
        ("测试集", config.TEST_CSV),
    ]:
        if Path(p).exists():
            n = sum(1 for _ in open(p, "r", encoding="utf-8"))
            print(f"    [OK ] {name}: {n} 条  {p}")
        else:
            print(f"    [缺失] {name}: 请先运行  python main.py build-data")

    # ---------- 4. 模型能否构建 ----------
    print("\n[4] 模型构建自检")
    try:
        from src.models.stcnet import Mynet

        cfg = Config()
        model = Mynet(cfg, pretrained_path=config.SWIN_PRETRAINED)
        n_param = sum(p.numel() for p in model.parameters())

        # 用假输入跑一次前向
        model.eval()
        with torch.no_grad():
            _, probs = model((torch.randn(1, 1, 224, 224), torch.randn(1, 1, 10)))

        print(f"    [OK ] Mynet 构建成功，参数量 {n_param:,}")
        print(f"    [OK ] 前向输出形状 {tuple(probs.shape)}（期望 (1, 2)）")
    except Exception as e:
        all_ok = False
        print(f"    [失败] 模型构建出错: {type(e).__name__}: {e}")

    # ---------- 汇总 ----------
    print()
    print("=" * 70)
    print("自检结论: " + ("全部就绪，可以开始训练" if all_ok else "存在问题，请按上面的提示处理"))
    print("=" * 70)
    return 0 if all_ok else 1


# ============================================================
# 阶段一：构建数据索引
# ============================================================


def cmd_build_data(args):
    """
    运行数据构建阶段：从 SOLAR-STORM2 原始数据生成三份索引 CSV。
    """
    from src.data.preprocess import run_preprocess_stage

    summary = run_preprocess_stage()
    return 0 if summary["train"] > 0 else 1


# ============================================================
# 阶段二：训练
# ============================================================


def cmd_train(args):
    """
    运行训练阶段。

    流程：
        1. 构建 Config（命令行参数可覆盖）
        2. 构建 DataLoader
        3. 构建模型（加载 Swin 预训练权重）
        4. 训练（含早停、最优权重保存）
        5. 画训练曲线
    """
    from src.data.dataset import get_dataloaders
    from src.models.stcnet import Mynet
    from src.train.trainer import set_seed, get_device, train
    from src.utils.visualization import plot_training_curves, save_metrics_json
    from src.utils.paths import ensure_dir

    # ---------- 1. 配置 ----------
    cfg = Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_supcon:
        cfg.use_supcon = False
    if args.no_weighted_sampler:
        cfg.use_weighted_sampler = False

    tag = args.tag or "stcnet"

    print("=" * 70)
    print(f"训练配置: {cfg}")
    print(f"实验标签: {tag}")
    print("=" * 70)

    # ---------- 2. 随机种子与设备 ----------
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"设备: {device}")

    # ---------- 3. 数据 ----------
    if not Path(config.TRAIN_CSV).exists():
        print(f"[错误] 找不到训练集索引 {config.TRAIN_CSV}")
        print("       请先运行: python main.py build-data")
        return 1

    loaders = get_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        img_size=cfg.img_size,
        use_weighted_sampler=cfg.use_weighted_sampler,
        # 冒烟测试时截断训练集
        max_train_samples=(200 if args.smoke else None),
    )

    if args.smoke:
        cfg.num_epochs = 1
        cfg.patience = 999          # 冒烟测试不早停
        print("[SMOKE] 冒烟测试模式：训练集截断为 200 条，只跑 1 个 epoch")

    # ---------- 4. 模型 ----------
    model = Mynet(
        cfg,
        pretrained_path=(None if args.no_pretrain else config.SWIN_PRETRAINED),
    )
    model = model.to(device)

    # ---------- 5. 训练 ----------
    ckpt_path = ensure_dir(config.CHECKPOINTS_DIR) / f"{tag}_best.pth"
    history, best_state = train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg,
        device=device,
        ckpt_path=ckpt_path,
    )

    # ---------- 6. 画训练曲线 ----------
    try:
        plot_training_curves(history, tag=tag)
    except Exception as e:
        print(f"[WARN] 画训练曲线失败: {e}")

    # 训练历史存 CSV，便于后续画图对比
    try:
        import pandas as pd

        hist_csv = ensure_dir(config.LOGS_DIR) / f"{tag}_history.csv"
        pd.DataFrame(history).to_csv(hist_csv, index=False, encoding="utf-8-sig")
        print(f"[记录] 训练历史已保存: {hist_csv}")
    except Exception as e:
        print(f"[WARN] 保存训练历史失败: {e}")

    print(f"\n最优权重已保存到: {ckpt_path}")
    print("接下来可以运行: python main.py test")
    return 0


# ============================================================
# 阶段三：测试集评估
# ============================================================


def cmd_test(args):
    """
    在测试集上评估模型，输出论文用的全部指标，并画图。

    权重来源（按优先级）：
        --ckpt 指定的路径  →  否则找 experiments/checkpoints/{tag}_best.pth
    """
    from src.data.dataset import SolarFlareDataset, build_transform, compute_param_range
    from src.models.stcnet import Mynet
    from src.train.evaluate import evaluate, print_metrics, sweep_thresholds
    from src.train.trainer import get_device
    from src.utils.visualization import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_prediction_samples,
        save_metrics_json,
    )
    from torch.utils.data import DataLoader

    cfg = Config()
    tag = args.tag or "stcnet"
    device = get_device(cfg.device)

    # ---------- 1. 找权重 ----------
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = config.CHECKPOINTS_DIR / f"{tag}_best.pth"

    if not ckpt_path.exists():
        print(f"[错误] 找不到权重文件: {ckpt_path}")
        print("       请先训练，或用 --ckpt 指定权重路径")
        return 1

    print("=" * 70)
    print(f"测试集评估 | 权重: {ckpt_path}")
    print("=" * 70)

    # ---------- 2. 模型 ----------
    # 加载时不传 pretrained_path，因为马上要用训练好的权重覆盖
    model = Mynet(cfg, pretrained_path=None)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 兼容两种保存格式：
    #   1. torch.save(model.state_dict())      → 直接就是 state_dict（本项目 trainer 用的）
    #   2. torch.save(model)                   → 是完整模型对象（原代码用的，如 model-29.pth）
    if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
        # 是完整模型对象被 pickle 的情况（state_dict 的值一定是 tensor）
        print("[权重] 检测到完整模型对象，提取其 state_dict")
        state = state.state_dict()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[权重] 缺失的键 {len(missing)} 个: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[权重] 多余的键 {len(unexpected)} 个: {unexpected[:5]}")
    model = model.to(device)
    model.eval()

    # ---------- 3. 测试集 DataLoader ----------
    if not Path(config.TEST_CSV).exists():
        print(f"[错误] 找不到测试集索引 {config.TEST_CSV}，请先运行 build-data")
        return 1

    param_min, param_max = compute_param_range(config.TRAIN_CSV)
    test_ds = SolarFlareDataset(
        config.TEST_CSV,
        img_size=cfg.img_size,
        param_min=param_min,
        param_max=param_max,
        transform=build_transform("test", cfg.img_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    print(f"\n测试集样本数: {len(test_ds)}")

    # ---------- 4. 评估 ----------
    metrics = evaluate(model, test_loader, device, threshold=cfg.eval_threshold)
    print_metrics(metrics, f"测试集评估结果（阈值={cfg.eval_threshold}）- {tag}")

    # ---------- 5. 画图 ----------
    try:
        plot_confusion_matrix(metrics["confusion"], tag=tag)
        plot_roc_curve(metrics["probs"], metrics["labels"], metrics["auc"], tag=tag)
        plot_prediction_samples(model, test_loader, device, tag=tag, n=8)
    except Exception as e:
        print(f"[WARN] 画图失败: {e}")

    save_metrics_json(metrics, tag=tag)

    # ---------- 6. 阈值扫描 ----------
    if args.sweep:
        sweep_thresholds(model, test_loader, device, thresholds=cfg.threshold_sweep)

    return 0


# ============================================================
# 一条龙
# ============================================================


def cmd_all(args):
    """
    一条龙：build-data → train → test。

    注意：正式训练很慢（全量 8000 条 × 50 epoch，Swin-Tiny 在 GPU 上也需要数小时），
         建议先跑 `python main.py train --smoke` 确认流程通，再跑 all。
    """
    print("#" * 70)
    print("# 步骤 1/3：构建数据索引")
    print("#" * 70)
    rc = cmd_build_data(args)
    if rc != 0:
        return rc

    print()
    print("#" * 70)
    print("# 步骤 2/3：训练")
    print("#" * 70)
    rc = cmd_train(args)
    if rc != 0:
        return rc

    print()
    print("#" * 70)
    print("# 步骤 3/3：测试集评估")
    print("#" * 70)
    return cmd_test(args)


# ============================================================
# 命令行解析
# ============================================================


def build_parser():
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="STCNet 太阳耀斑预报项目主入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    sub.add_parser("check", help="检查环境与数据是否就绪")

    # build-data
    p_build = sub.add_parser("build-data", help="阶段一：构建索引 CSV")
    p_build.add_argument("--no-downsample", action="store_true",
                         help="不对负样本下采样（保留全部负样本）")

    # train
    p_train = sub.add_parser("train", help="阶段二：训练模型")
    p_train.add_argument("--epochs", type=int, default=None, help="epoch 数")
    p_train.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    p_train.add_argument("--tag", type=str, default=None, help="实验标签")
    p_train.add_argument("--no-supcon", action="store_true", help="关闭对比学习（消融实验）")
    p_train.add_argument("--no-pretrain", action="store_true", help="不加载预训练权重（消融实验）")
    p_train.add_argument("--no-weighted-sampler", action="store_true", help="关闭加权采样")
    p_train.add_argument("--smoke", action="store_true", help="冒烟测试：小样本 1 个 epoch")

    # test
    p_test = sub.add_parser("test", help="阶段三：测试集评估")
    p_test.add_argument("--ckpt", type=str, default=None, help="权重文件路径")
    p_test.add_argument("--tag", type=str, default=None, help="实验标签")
    p_test.add_argument("--sweep", action="store_true", help="额外做阈值扫描")

    # all
    p_all = sub.add_parser("all", help="一条龙：build-data → train → test")
    p_all.add_argument("--epochs", type=int, default=None, help="epoch 数")
    p_all.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    p_all.add_argument("--tag", type=str, default=None, help="实验标签")
    p_all.add_argument("--no-downsample", action="store_true", help="不对负样本下采样")
    p_all.add_argument("--no-supcon", action="store_true", help="关闭对比学习")
    p_all.add_argument("--no-pretrain", action="store_true", help="不加载预训练权重")
    p_all.add_argument("--no-weighted-sampler", action="store_true", help="关闭加权采样")
    p_all.add_argument("--smoke", action="store_true", help="冒烟测试：小样本 1 个 epoch")
    p_all.add_argument("--ckpt", type=str, default=None, help="（test 步骤用）权重路径")
    p_all.add_argument("--sweep", action="store_true", help="（test 步骤用）阈值扫描")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 分发到对应阶段
    if args.command == "check":
        return cmd_check(args)
    if args.command == "build-data":
        if args.no_downsample:
            config.DOWNSAMPLE_NEGATIVE = False
        return cmd_build_data(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "test":
        return cmd_test(args)
    if args.command == "all":
        if args.no_downsample:
            config.DOWNSAMPLE_NEGATIVE = False
        return cmd_all(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
