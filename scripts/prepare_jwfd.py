"""
======================================================================
scripts/prepare_jwfd.py
离线预缩：把 JW-FD 的 600×600 PNG 批量转成 224×224 JPEG
======================================================================

为什么需要这一步
----------------
JW-FD 的原始图片是 600×600 PNG（约 120 KB/张）。而我们的模型输入是 224×224，
所以**每读一张都要做一次 600→224 的缩放**，这个开销完全发生在 CPU 上。

实测（本机 RTX 5070 Laptop + 16 逻辑核）：
    训练  158,935 张 600×600 PNG → 一个 epoch 约 17 分钟（GPU 前向+反向）
    验证   18,262 张 600×600 PNG → 约 8 分钟（主要是 CPU 解码 + 缩放）
    → 合计约 25 分钟/epoch

预缩之后：
    - 单张从 ~120 KB 降到 ~7 KB（实测），读盘量减少约 94%
    - 训练时**不再需要 Resize**，CPU 开销基本消失
    - 预计 epoch 时间降到 ~18 分钟，且 5 种子实验的总耗时会显著下降

副作用
------
    - 占用额外磁盘：200,938 × 7 KB ≈ **1.34 GB**（实测）
    - 改了输入尺寸或想换 Bth 档位，需要重跑本脚本

用法
----
    # 全量转换（约 1.34 GB；实测 4 进程约 128 张/秒 → 全量约 26 分钟，8 进程更快）
    python scripts/prepare_jwfd.py

    # 先小规模试跑，确认无误
    python scripts/prepare_jwfd.py --limit 2000

    # 自定义
    python scripts/prepare_jwfd.py --size 224 --quality 92 --workers 8

    # 只转换某个 split
    python scripts/prepare_jwfd.py --splits train

转换完成后
----------
    1. 把 config.py 里的 USE_PREPROCESSED_PNG 改成 True
    2. 重新跑 python main.py train --smoke 验证
    3. 确认无误后，如需省空间可考虑精简原始 PNG（建议先保留，确认稳定再动）

注意
----
本脚本用 multiprocessing 并行，Windows 下必须有 if __name__ == "__main__" 保护，
否则子进程会重新导入主模块并递归启动（本项目已加保护）。
======================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# 让脚本能从项目根目录 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

import config

# 关闭 PIL 对超大图片的警告（磁图都是 600×600，安全）
Image.MAX_IMAGE_PIXELS = None


# ======================================================================
# 单张转换（在子进程中执行）
# ======================================================================

def convert_one(task: tuple[str, str, int, int]) -> tuple[str, bool, str]:
    """
    转换单张图片：600×600 PNG → size×size JPEG。

    Parameters
    ----------
    task : tuple
        (src_path, dst_path, size, quality)

    Returns
    -------
    (dst_path, ok, msg) : tuple[str, bool, str]
    """
    src, dst, size, quality = task
    try:
        # 已存在就跳过（便于断点续跑）
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst, True, "skip"

        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as im:
            # JW-FD 的 PNG 已经是 mode='L' 单通道灰度；万一不是就转灰度
            if im.mode != "L":
                im = im.convert("L")
            # 已经是目标尺寸就不用缩放（省时间）
            if im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            im.save(dst_p, "JPEG", quality=quality, optimize=True)

        return dst, True, "ok"
    except Exception as e:                                        # noqa: BLE001
        return dst, False, f"{type(e).__name__}: {e}"


# ======================================================================
# 收集待转换列表
# ======================================================================

def collect_tasks(splits: list[str], size: int, quality: int, limit: int | None) -> list[tuple]:
    """
    从三份 CSV 里收集所有需要的图片，生成转换任务列表。

    去重：同一张图可能出现在多个标签列/多个 split 里？不会——
        每个样本只属于一个 split，但为了保险仍用 set 去重。
    """
    csv_map = {
        "train": config.TRAIN_CSV,
        "val": config.VAL_CSV,
        "test": config.TEST_CSV,
    }

    seen: set[str] = set()
    tasks: list[tuple] = []

    for split in splits:
        csv_path = csv_map[split]
        if not Path(csv_path).exists():
            print(f"  [跳过] {split}: {csv_path} 不存在")
            continue

        # 只读两列，很快
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in {config.COL_IMAGE_PATH, config.COL_IMAGE_FILENAME},
        )
        n = len(df)
        print(f"  {split:5s}: {n:>7,} 行  ({Path(csv_path).name})")

        paths = (df[config.COL_IMAGE_PATH].astype(str).tolist()
                 if config.COL_IMAGE_PATH in df.columns else [""] * n)
        fns = (df[config.COL_IMAGE_FILENAME].astype(str).tolist()
               if config.COL_IMAGE_FILENAME in df.columns else [None] * n)

        for rel, fn in zip(paths, fns):
            if limit is not None and len(tasks) >= limit:
                break

            # 源文件绝对路径（复用 dataset 里的解析逻辑，保证一致）
            from src.data.dataset import resolve_image_path
            src = resolve_image_path(rel, fn)

            if src in seen:
                continue
            seen.add(src)

            # 输出路径：png_224/{AR}/{filename 去扩展名}.jpg
            fname = fn if fn else Path(src).name
            stem = Path(fname).stem                      # 去掉 .png
            ar = stem.split("_")[0]                      # 'AR3947'
            if ar.upper().startswith("AR"):
                ar = ar[2:]                              # '3947'
            dst = str(config.JWFD_PNG_224_DIR / ar / f"{stem}.jpg")

            tasks.append((src, dst, size, quality))

    return tasks


# ======================================================================
# main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="JW-FD 离线预缩：600×600 PNG → 224×224 JPEG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--size", type=int, default=config.IMG_SIZE,
                    help=f"目标边长（默认 {config.IMG_SIZE}）")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量（默认 92）")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="并行进程数（默认 8）")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"], help="要处理的 split")
    ap.add_argument("--limit", type=int, default=None,
                    help="最多转换多少张（调试用）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已存在的输出文件，强制重转")
    args = ap.parse_args()

    print("=" * 78)
    print("  JW-FD 离线预缩")
    print("=" * 78)
    print(f"  源目录    : {config.JWFD_PNG_DIR}")
    print(f"  输出目录  : {config.JWFD_PNG_224_DIR}")
    print(f"  目标尺寸  : {args.size}×{args.size}  JPEG q={args.quality}")
    print(f"  并行进程  : {args.workers}")
    print(f"  splits    : {args.splits}")
    if args.limit:
        print(f"  上限      : {args.limit} 张（调试模式）")
    print()

    # ---------- 1. 收集任务 ----------
    print("[1] 收集待转换图片")
    tasks = collect_tasks(args.splits, args.size, args.quality, args.limit)
    print(f"  合计 {len(tasks):,} 张（已去重）")

    if not tasks:
        print("  [错误] 没有待转换的图片，请检查 CSV 路径")
        return 1

    # ---------- 2. 强制重转：删掉已有输出 ----------
    if args.force:
        print("\n[2] --force：清理已有输出")
        removed = 0
        for _src, dst, _s, _q in tasks:
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                    removed += 1
                except OSError:
                    pass
        print(f"  已删除 {removed:,} 个旧文件")
    else:
        print("\n[2] 跳过已存在的输出文件（断点续跑）")

    # ---------- 3. 开始转换 ----------
    print(f"\n[3] 开始转换（{args.workers} 进程）")
    t0 = time.time()
    ok_n = skip_n = fail_n = 0
    fails: list[str] = []
    done = 0

    with Pool(processes=args.workers) as pool:
        # imap_unordered 边算边返回，便于显示进度
        for dst, ok, msg in pool.imap_unordered(convert_one, tasks, chunksize=64):
            done += 1
            if ok:
                if msg == "skip":
                    skip_n += 1
                else:
                    ok_n += 1
            else:
                fail_n += 1
                if len(fails) < 10:
                    fails.append(f"{Path(dst).name}: {msg}")

            # 进度
            if done % 2000 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done:>7,}/{len(tasks):,}"
                      f"  新增 {ok_n:,}  跳过 {skip_n:,}  失败 {fail_n:,}"
                      f"  | {rate:6.0f} 张/秒 | 剩余约 {eta/60:5.1f} 分钟")

    elapsed = time.time() - t0
    print()
    print("=" * 78)
    print(f"  完成：新增 {ok_n:,}，跳过 {skip_n:,}，失败 {fail_n:,}")
    print(f"  总耗时 {elapsed/60:.1f} 分钟（{len(tasks)/max(elapsed,1e-6):.0f} 张/秒）")
    if fails:
        print("  失败示例：")
        for f in fails:
            print(f"    {f}")
    print("=" * 78)

    # ---------- 4. 体积与下一步 ----------
    out_dir = config.JWFD_PNG_224_DIR
    if out_dir.exists():
        total = sum(f.stat().st_size for f in out_dir.rglob("*.jpg"))
        n_files = sum(1 for _ in out_dir.rglob("*.jpg"))
        print(f"\n  输出目录: {out_dir}")
        print(f"  文件数  : {n_files:,}")
        print(f"  总体积  : {total/1024**3:.2f} GB")
        print(f"  平均单张: {total/max(n_files,1)/1024:.1f} KB")

    print()
    print("下一步：")
    print("  1. 修改 config.py：")
    print("         USE_PREPROCESSED_PNG = True")
    print("  2. 验证：")
    print("         python main.py train --smoke")
    print("  3. 确认无误后对比 epoch 耗时（应明显缩短）")

    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
