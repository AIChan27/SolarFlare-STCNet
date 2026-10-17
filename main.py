"""
----------------------------------------------------------------------
main.py
项目主入口，按阶段调用 pipeline

用法（在项目根目录运行）：
    python main.py check                     # 检查环境是否就绪、模型能否构建
    python main.py check-data                # 校验 JW-FD 数据集（行数/列/标签/图片/划分）
    python main.py check-data --sample 500   # 抽样更多图片做存在性检查
    python main.py train                     # 阶段二：训练
    python main.py train --smoke             # 冒烟测试（小样本 1 个 epoch）
    python main.py test --ckpt <权重路径>     # 阶段三：测试集评估
    python main.py test --sweep              # 评估 + 阈值扫描
    python main.py all                       # 一条龙：check-data → train → test

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

    # ---------- 2. 数据文件（JW-FD v1，10% 子集）----------
    print("\n[2] 数据（JW-FD v1，约 10% 代表性子集）")
    paths = {
        "JW-FD 根目录": config.JWFD_DIR,
        "PNG 磁图目录": config.JWFD_PNG_DIR,
        "耀斑事件表": config.JWFD_FLARE_LABELS,
        "Swin 预训练权重": config.SWIN_PRETRAINED,
    }
    all_ok = True
    for name, p in paths.items():
        ok = Path(p).exists()
        all_ok &= ok
        print(f"    [{'OK ' if ok else '缺失'}] {name:16s} {p}")

    # 活动区数与 PNG 总数核对（10% 子集应为 307 个活动区 / 200,938 张）
    if Path(config.JWFD_PNG_DIR).exists():
        print("    ... 正在统计 PNG 数量（约需几秒）")
        ar_dirs = [d for d in Path(config.JWFD_PNG_DIR).iterdir() if d.is_dir()]
        n_png = sum(1 for _ in Path(config.JWFD_PNG_DIR).rglob("*.png"))
        flag = "OK " if (len(ar_dirs) == 307 and n_png == 200938) else "注意"
        print(f"    [{flag}] 活动区 {len(ar_dirs):,} 个，PNG {n_png:,} 张"
              f"（10% 子集预期 307 / 200,938）")

    # ---------- 3. 数据集 CSV ----------
    print("\n[3] 数据集 CSV（66 列；官方已按活动区 8:1:1 划分，无需自己切分）")
    for name, p, expect in [
        ("训练集", config.TRAIN_CSV, 158935),
        ("验证集", config.VAL_CSV, 18262),
        ("测试集", config.TEST_CSV, 23741),
    ]:
        if Path(p).exists():
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                n = sum(1 for _ in f) - 1          # 减去表头
            flag = "OK " if n == expect else "注意"
            print(f"    [{flag}] {name}: {n:>7,} 条（预期 {expect:,}）  {Path(p).name}")
        else:
            print(f"    [缺失] {name}: {p}")
            all_ok = False

    print(f"\n    主任务标签列 : {config.LABEL_COL}")
    print(f"    特征维度     : {config.NUM_PARAMS}（config.FEATURE_COLS）")

    # ---------- 4. 模型能否构建 ----------
    print("\n[4] 模型构建自检")
    try:
        from src.models.stcnet import Mynet

        cfg = Config()
        model = Mynet(cfg, pretrained_path=config.SWIN_PRETRAINED)
        n_param = sum(p.numel() for p in model.parameters())

        # 用假输入跑一次前向（特征维必须等于 config.NUM_PARAMS，不能写死 10）
        model.eval()
        with torch.no_grad():
            _, probs = model(
                (torch.randn(1, 1, 224, 224), torch.randn(1, 1, config.NUM_PARAMS))
            )

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
# 数据完整性校验（替代原来的 build-data）
# ============================================================


def cmd_validate_data(args):
    """
    校验 JW-FD 数据集是否可直接用于训练。

    为什么不再需要 build-data
    -------------------------
    SOLAR-STORM2 时代，需要我们自己"按时间切分 + 负样本下采样 + 生成三联 CSV"。
    JW-FD 官方已经用 split_dataset.py 按**活动区**做了 8:1:1 分层划分（seed=62），
    并直接提供了 train / val / test 三份 66 列 CSV。
    所以我们只需**校验**，不再**构建**——src/data/preprocess.py 因此作废。

    校验内容
    --------
        1. 三份 CSV 是否存在、行数是否与官方一致
        2. 66 列是否齐全；config.FEATURE_COLS 是否全部命中；LABEL_COL 是否存在
        3. 每个标签列的正样本数与缺失值
        4. 抽样检查图片路径是否真实存在（默认 200 张）
        5. 活动区级划分是否真的没有交叉（同一 AR 不跨 train/val/test）
    """
    import csv as _csv
    import random as _random

    print("=" * 76)
    print("JW-FD 数据集校验")
    print("=" * 76)
    print(f"  JW-FD 版本: v1（约 10% 子集）")
    print(f"  根目录    : {config.JWFD_DIR}")
    print(f"  PNG 目录  : {config.JWFD_PNG_DIR}")
    print(f"  主任务标签: {config.LABEL_COL}")
    print()

    # ---------- 1. 三份 CSV 行数 ----------
    expect = {"train": 158935, "val": 18262, "test": 23741}
    files = {"train": config.TRAIN_CSV, "val": config.VAL_CSV, "test": config.TEST_CSV}
    headers: dict[str, list[str]] = {}
    ok_all = True

    print("[1] CSV 文件与行数")
    for split, p in files.items():
        if not Path(p).exists():
            print(f"    [缺失] {split:5s}: {p}")
            ok_all = False
            continue
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            rd = _csv.reader(f)
            try:
                hdr = next(rd)
            except StopIteration:
                print(f"    [空文件] {split}: {p}")
                ok_all = False
                continue
            headers[split] = [h.strip() for h in hdr]
            n = sum(1 for _ in rd)
        flag = "OK  " if n == expect[split] else "注意"
        print(f"    [{flag}] {split:5s}: {n:>7,} 行 / {len(headers[split])} 列"
              f"（预期 {expect[split]:,} 行）")

    # ---------- 2. 列完整性 ----------
    print("\n[2] 列完整性检查")
    if headers:
        any_hdr = headers[next(iter(headers))]
        missing_feat = [c for c in config.FEATURE_COLS if c not in any_hdr]
        has_label = config.LABEL_COL in any_hdr
        n_label_cols = sum(1 for c in any_hdr if c.startswith("flare_label_"))
        n_class_cols = sum(1 for c in any_hdr if c.startswith("flare_class_"))

        print(f"    总列数            : {len(any_hdr)}（论文口径 66）")
        print(f"    二分类标签列数    : {n_label_cols}（预期 28 = 7 窗口 × 4 阈值）")
        print(f"    类别标签列数      : {n_class_cols}（预期 7）")
        print(f"    29 个特征列缺失   : {missing_feat if missing_feat else '无'}")
        print(f"    主任务标签 {config.LABEL_COL}: {'存在' if has_label else '缺失'}")
        if missing_feat or not has_label:
            ok_all = False

    # ---------- 3. 标签分布 ----------
    print("\n[3] 标签分布（训练集）")
    if "train" in headers:
        hdr = headers["train"]
        label_cols = [c for c in hdr if c.startswith("flare_label_")]
        # 只统计主任务 + 几个常用标签，避免扫 28 遍
        watch = [config.LABEL_COL] + [
            c for c in ("flare_label_C1.0_24hr", "flare_label_M1.0_24hr",
                        "flare_label_M5.0_48hr", "flare_label_X1.0_24hr")
            if c in label_cols and c != config.LABEL_COL
        ]
        idxs = {c: hdr.index(c) for c in watch}
        counts = {c: [0, 0] for c in watch}      # [负, 正]
        with open(files["train"], "r", encoding="utf-8", errors="replace", newline="") as f:
            rd = _csv.reader(f)
            next(rd)
            for row in rd:
                for c, i in idxs.items():
                    if i < len(row):
                        v = row[i].strip()
                        if v == "1":
                            counts[c][1] += 1
                        elif v == "0":
                            counts[c][0] += 1
        for c in watch:
            neg, pos = counts[c]
            tot = neg + pos
            mark = "  ← 主任务" if c == config.LABEL_COL else ""
            print(f"    {c:<26s} 正 {pos:>7,} / 负 {neg:>8,}"
                  f"（正样本率 {pos / max(tot,1):>6.2%}）{mark}")

    # ---------- 4. 抽样检查图片存在性 ----------
    print("\n[4] 图片路径抽样检查")
    if "train" in headers:
        hdr = headers["train"]
        i_path = hdr.index(config.COL_IMAGE_PATH) if config.COL_IMAGE_PATH in hdr else None
        i_fn = hdr.index(config.COL_IMAGE_FILENAME) if config.COL_IMAGE_FILENAME in hdr else None
        if i_path is not None or i_fn is not None:
            from src.data.dataset import resolve_image_path
            rows = []
            with open(files["train"], "r", encoding="utf-8", errors="replace", newline="") as f:
                rd = _csv.reader(f)
                next(rd)
                for k, row in enumerate(rd):
                    if k >= 2000:
                        break
                    rows.append(row)
            sample = _random.Random(0).sample(
                rows, min(getattr(args, "sample", 200), len(rows))
            )
            miss, ok_n = [], 0
            for row in sample:
                rel = row[i_path] if i_path is not None and i_path < len(row) else ""
                fn = row[i_fn] if i_fn is not None and i_fn < len(row) else None
                p = resolve_image_path(rel, fn)
                if Path(p).exists():
                    ok_n += 1
                elif len(miss) < 3:
                    miss.append(p)
            print(f"    抽样 {len(sample)} 张 → 存在 {ok_n}，缺失 {len(sample) - ok_n}")
            for m in miss:
                print(f"      缺失示例: {m}")
            if ok_n < len(sample):
                ok_all = False

    # ---------- 5. 活动区划分交叉检查 ----------
    print("\n[5] 活动区（AR）级划分交叉检查")
    ar_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        if split not in headers or config.COL_IMAGE_FILENAME not in headers[split]:
            continue
        i_fn = headers[split].index(config.COL_IMAGE_FILENAME)
        s: set[str] = set()
        with open(files[split], "r", encoding="utf-8", errors="replace", newline="") as f:
            rd = _csv.reader(f)
            next(rd)
            for row in rd:
                if i_fn < len(row):
                    ar = row[i_fn].split("_")[0]      # 'AR3947'
                    s.add(ar)
        ar_sets[split] = s
        print(f"    {split:5s}: {len(s)} 个活动区")

    if len(ar_sets) == 3:
        overlaps = {
            "train∩val": ar_sets["train"] & ar_sets["val"],
            "train∩test": ar_sets["train"] & ar_sets["test"],
            "val∩test": ar_sets["val"] & ar_sets["test"],
        }
        clean = all(len(v) == 0 for v in overlaps.values())
        for k, v in overlaps.items():
            print(f"    {k:<12s}: {len(v)} 个重叠"
                  + (f"  ← 泄漏！{sorted(v)[:5]}" if v else "  ✓"))
        if clean:
            print("    ✓ 活动区级划分无交叉，不存在同一活动区跨分区的泄漏")
        else:
            ok_all = False

    # ---------- 汇总 ----------
    print()
    print("=" * 76)
    print("校验结论: " + ("全部通过，可以开始训练" if ok_all else "存在问题，请按上面的提示处理"))
    print("=" * 76)
    return 0 if ok_all else 1


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

    # ★ 混合精度：--amp 启用；--amp-dtype 覆盖类型
    if getattr(args, "amp", False):
        cfg.use_amp = True
    if getattr(args, "amp_dtype", None):
        cfg.amp_dtype = args.amp_dtype
    if cfg.use_amp:
        print(f"[AMP] 混合精度已启用：dtype={cfg.amp_dtype}")

    # ★ 新增：覆盖随机种子
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed

    tag = args.tag or "stcnet"
    # ★ 新增：指定了 seed 就自动拼到 tag 后，保证每个种子的产物互不覆盖
    #    例如 --tag stcnet --seed 1  →  tag = "stcnet_seed1"
    #         → checkpoints/stcnet_seed1_best.pth, logs/stcnet_seed1_history.csv
    if getattr(args, "seed", None) is not None:
        tag = f"{tag}_seed{args.seed}"

    print("=" * 70)
    print(f"训练配置: {cfg}")
    print(f"实验标签: {tag}")
    print(f"随机种子: {cfg.seed}")          # ★ 新增一行，便于日志核对
    print("=" * 70)

    # ---------- 2. 随机种子与设备 ----------
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"设备: {device}")

    # ★ 新增：DataLoader 子进程的种子函数（保证数据增强也可复现）
    from functools import partial
    from src.train.trainer import set_seed, get_device, train, set_worker_seed
    worker_init_fn = partial(set_worker_seed)

    # ---------- 3. 数据 ----------
    if not Path(config.TRAIN_CSV).exists():
        print(f"[错误] 找不到训练集索引 {config.TRAIN_CSV}")
        print("       请先检查数据集: python main.py check-data")
        return 1

    loaders = get_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        img_size=cfg.img_size,
        use_weighted_sampler=cfg.use_weighted_sampler,
        # 冒烟测试时截断训练集
        max_train_samples=(200 if args.smoke else None),
        worker_init_fn=worker_init_fn,          # ★ 新增
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

    # ★ 新增：覆盖随机种子
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed

    tag = args.tag or "stcnet"
    # ★ 新增：指定了 seed 就自动拼到 tag 后，保证每个种子的产物互不覆盖
    #    例如 --tag stcnet --seed 1  →  tag = "stcnet_seed1"
    #         → checkpoints/stcnet_seed1_best.pth, logs/stcnet_seed1_history.csv
    if getattr(args, "seed", None) is not None:
        tag = f"{tag}_seed{args.seed}"

    print(f"实验标签: {tag}")    # 建议加上这行，方便核对

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
        print(f"[错误] 找不到测试集 CSV {config.TEST_CSV}，请先运行 check-data")
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

    # ---------- 4. 确定分类阈值 ----------
    # 两种来源：
    #   (a) 默认：config.EVAL_THRESHOLD（固定 0.5）
    #   (b) --val-threshold：在【验证集】上扫描并选最优阈值
    #
    # 为什么推荐 (b)：阈值是一个"被选出来的超参数"。如果直接拿测试集扫阈值
    # 再挑最好的，等于用测试集答案挑一条最有利的分割线 —— 这是测试集泄漏，
    # 会让报告指标虚高。实测该泄漏可达 0.04，比我们要验证的模型改进还大。
    chosen_threshold = float(cfg.eval_threshold)
    val_scan = None
    use_val_threshold = bool(getattr(args, "val_threshold", False))

    if use_val_threshold:
        if not Path(config.VAL_CSV).exists():
            print(f"[错误] --val-threshold 需要验证集 CSV，但找不到: {config.VAL_CSV}")
            return 1
        from src.train.evaluate import select_threshold

        val_ds = SolarFlareDataset(
            config.VAL_CSV,
            img_size=cfg.img_size,
            param_min=param_min,
            param_max=param_max,
            transform=build_transform("val", cfg.img_size),
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        print(f"\n验证集样本数: {len(val_ds):,}（仅用于选阈值）")

        chosen_threshold, best_row, val_scan = select_threshold(
            model, val_loader, device,
            metric=(getattr(args, "threshold_metric", None) or "tss"),
            thresholds=cfg.threshold_sweep,
        )
        del val_loader
    else:
        print(f"\n[提示] 当前用固定阈值 {chosen_threshold:.2f}。")
        print("       若要避免测试集泄漏，建议加 --val-threshold（在验证集上选阈值）。")

    # ---------- 5. 测试集评估（只评估一次，阈值已定死）----------
    metrics = evaluate(model, test_loader, device, threshold=chosen_threshold)
    source = "验证集选定" if use_val_threshold else "固定值"
    print_metrics(
        metrics,
        f"测试集评估结果（阈值={chosen_threshold:.2f}，{source}）- {tag}",
    )
    if use_val_threshold:
        print(f">>> 该阈值在验证集上选定后固定，测试集未参与任何阈值搜索（无泄漏）")
    metrics["threshold_source"] = "val" if use_val_threshold else "fixed"

    # ---------- 6. 画图 ----------
    try:
        plot_confusion_matrix(metrics["confusion"], tag=tag)
        plot_roc_curve(metrics["probs"], metrics["labels"], metrics["auc"], tag=tag)
        plot_prediction_samples(model, test_loader, device, tag=tag, n=8)
    except Exception as e:
        print(f"[WARN] 画图失败: {e}")

    save_metrics_json(metrics, tag=tag)

    # ---------- 7. 诊断用的测试集阈值扫描 ----------
    if args.sweep:
        print()
        print("[诊断] 以下扫描在【测试集】上进行，仅供观察趋势，")
        print("       其中的'最优点'不能用于报告 —— 报告请用上面那个阈值。")
        sweep_thresholds(model, test_loader, device, thresholds=cfg.threshold_sweep)

    return 0


# ============================================================
# 一条龙
# ============================================================


def cmd_all(args):
    """
    一条龙：check-data → train → test。

    ⚠️ 耗时提醒（JW-FD 规模下）：
        训练集 158,935 条、batch 32 → 一个 epoch 约 18 分钟（含验证集评估）。
        默认 NUM_EPOCHS=12 / PATIENCE=5，因此：
            最快：约 6 个 epoch 早停  → 约 1.8 小时
            最慢：跑满 12 个 epoch    → 约 3.6 小时
        之后还会自动跑一遍测试集评估（23,741 条，约 10 分钟）。

        想先快速看个大概，可以加 --epochs 6。
        建议先跑 `python main.py train --smoke` 确认流程通。

    ⚠️ 产物命名：不传 --tag 时默认 tag='stcnet'，会覆盖旧的
        experiments/checkpoints/stcnet_best.pth（SOLAR-STORM2 时代的模型）。
        建议显式传 --tag jwfd 之类，把两代实验分开。
    """
    print("#" * 70)
    print("# 步骤 1/3：校验数据集")
    print("#" * 70)
    rc = cmd_validate_data(args)
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

    # check-data（替代原来的 build-data）
    p_check_data = sub.add_parser(
        "check-data",
        help="校验 JW-FD 数据集：行数/列完整性/标签分布/图片存在性/活动区划分",
    )
    p_check_data.add_argument("--sample", type=int, default=200,
                              help="抽查多少张图片验证存在性（默认 200）")

    # train
    p_train = sub.add_parser("train", help="阶段二：训练模型")
    p_train.add_argument("--epochs", type=int, default=None, help="epoch 数")
    p_train.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    p_train.add_argument("--tag", type=str, default=None, help="实验标签")
    p_train.add_argument("--no-supcon", action="store_true", help="关闭对比学习（消融实验）")
    p_train.add_argument("--no-pretrain", action="store_true", help="不加载预训练权重（消融实验）")
    p_train.add_argument("--no-weighted-sampler", action="store_true", help="关闭加权采样")
    p_train.add_argument("--smoke", action="store_true", help="冒烟测试：小样本 1 个 epoch")
    p_train.add_argument("--amp", action="store_true",
                         help="启用混合精度训练（bf16，约 1.5× 加速、显存降约 1 GB）")
    p_train.add_argument("--amp-dtype", type=str, default=None, choices=["bf16", "fp16"],
                         help="混合精度类型，默认跟随 config.AMP_DTYPE（bf16）")
    # ★ 新增：随机种子，用于多种子重复实验
    p_train.add_argument("--seed", type=int, default=None,
                         help="随机种子。不填则用 config.RANDOM_SEED；"
                              "填了会自动把 seed 拼到 tag 后（如 stcnet_seed1），避免覆盖已有结果")

    # test
    p_test = sub.add_parser("test", help="阶段三：测试集评估")
    p_test.add_argument("--ckpt", type=str, default=None, help="权重文件路径")
    p_test.add_argument("--tag", type=str, default=None, help="实验标签")
    p_test.add_argument("--sweep", action="store_true", help="额外做阈值扫描（诊断用）")
    p_test.add_argument("--val-threshold", action="store_true",
                        help="★推荐：在验证集上选阈值，再用它评估测试集（避免测试集泄漏）")
    p_test.add_argument("--threshold-metric", type=str, default=None,
                        choices=["tss", "f1", "precision", "accuracy"],
                        help="选阈值的判据，默认 tss")
    p_test.add_argument("--seed", type=int, default=None, help="随机种子（用于定位对应权重）")

    # all
    p_all = sub.add_parser("all", help="一条龙：check-data → train → test")
    p_all.add_argument("--epochs", type=int, default=None, help="epoch 数")
    p_all.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    p_all.add_argument("--tag", type=str, default=None, help="实验标签")
    # ★ 新增
    p_all.add_argument("--seed", type=int, default=None, help="随机种子")

    p_all.add_argument("--no-downsample", action="store_true", help="（已弃用，JW-FD 不做下采样）")
    p_all.add_argument("--no-supcon", action="store_true", help="关闭对比学习")
    p_all.add_argument("--no-pretrain", action="store_true", help="不加载预训练权重")
    p_all.add_argument("--no-weighted-sampler", action="store_true", help="关闭加权采样")
    p_all.add_argument("--smoke", action="store_true", help="冒烟测试：小样本 1 个 epoch")
    p_all.add_argument("--amp", action="store_true", help="启用混合精度（bf16，约 1.5×）")
    p_all.add_argument("--amp-dtype", type=str, default=None, choices=["bf16", "fp16"],
                       help="混合精度类型，默认 bf16")
    p_all.add_argument("--ckpt", type=str, default=None, help="（test 步骤用）权重路径")
    p_all.add_argument("--sweep", action="store_true", help="（test 步骤用）阈值扫描（诊断用）")
    p_all.add_argument("--val-threshold", action="store_true",
                       help="（test 步骤用）★推荐：验证集选阈值，避免测试集泄漏")
    p_all.add_argument("--threshold-metric", type=str, default=None,
                       choices=["tss", "f1", "precision", "accuracy"],
                       help="（test 步骤用）选阈值的判据，默认 tss")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 分发到对应阶段
    if args.command == "check":
        return cmd_check(args)
    if args.command == "check-data":
        return cmd_validate_data(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "test":
        return cmd_test(args)
    if args.command == "all":
        return cmd_all(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
