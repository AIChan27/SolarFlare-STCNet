"""
======================================================================
⚠️ 本模块已作废（DEPRECATED）—— 保留仅供查阅历史逻辑，请勿再调用
======================================================================

作废原因
--------
本模块是 **SOLAR-STORM2 时代**的数据构建脚本：官方只给了磁图 + 参数表，
没有划分训练/测试集，所以需要我们自己做「按时间切分 + 负样本下采样 +
生成三联 CSV（trainflare1/validflare1/testflare1）」。

现在项目已切换到 **JW-FD（JinWu-Flare Dataset）**，情况完全不同：
    - 官方 split_dataset.py 已经按**活动区**做了 8:1:1 分层划分（seed=62），
      并且直接提供了 train / val / test 三份 **66 列** CSV；
    - 标签也内置在 CSV 里（28 个二分类列 + 7 个类别列），不需要我们生成；
    - 图片是 600×600 PNG，路径由 CSV 的 image_path 列给出。

所以「切分 / 下采样 / 生成 CSV」这三件事**全部不需要了**。

现在的替代方案
--------------
    python main.py check-data     # 校验数据集（行数/列/标签/图片/活动区划分）

真正的数据读取逻辑在 src/data/dataset.py（按列名读取 66 列 CSV）。

如果将来要回到 SOLAR-STORM2 做对照实验，本文件仍可作为参考，但需要
配合当时的 config.py 版本使用（当前 config 已不再定义 TRAIN_PERIODS 等）。

======================================================================
以下为原始说明（历史存档）
======================================================================

src/data/preprocess.py
阶段一：从 SOLAR-STORM2 原始数据构建训练/验证/测试索引 CSV

本模块的作用：
    官方数据集只给了两样东西（见 docs/notes/dataset_build.md）：
        1. magnetogram_jpg/  73,810 张灰度磁图（平铺在一个目录里）
        2. para_flare.txt    73,810 行 = 文件名 + 10 个磁场参数 + 耀斑标签
    官方【没有】划分训练/测试集，也【没有】提供任何预处理脚本。
    本模块负责把原始数据变成 PyTorch 能直接吃的三份 CSV。

整体流程：
    1. read_para_flare      ：读 para_flare.txt，解析出 (文件名, 时间, 活动区号, 10参数, 标签)
    2. filter_by_periods    ：按时间段筛选（论文 4.1 节的时间划分）
    3. downsample_negative  ：负样本下采样（论文规则：每天每个活动区只留一条）
    4. attach_image_path    ：把文件名映射成真实图片绝对路径（.fits → .fits.jpg）
    5. write_split_csv      ：写出 CSV（第一列图片路径 + 中间 10 列参数 + 最后一列标签）
    6. run_preprocess_stage ：以上步骤的封装，作为 pipeline 的阶段入口

输出（三份 CSV，列结构完全相同，只有行不同）：
    data/index/trainflare1.csv
    data/index/validflare1.csv
    data/index/testflare1.csv

    列结构（13 列）：
        第 1 列      : 图片绝对路径
        第 2–11 列   : 10 个磁场特征参数（TOTUSJH, TOTPOT, ... R_VALUE）
        第 12 列     : 标签（1 = 48h 内发生 M 级及以上耀斑；0 = 没有）

    为什么是"参数列没有表头、标签在最后一列"这种朴素结构？
        因为它是原论文代码 src/data/dataset.py 里 My_Dataset 直接读取的格式，
        保持兼容可以让 DataLoader 一行都不用改。
        具体读取方式见 src/data/dataset.py。
----------------------------------------------------------------------
"""

# from __future__ import annotations
#   作用：让类型注解（如 list[dict]）在 Python 3.8 及以下也能用
from __future__ import annotations

# re：正则表达式，用于从文件名里抠出活动区编号和观测时间
import re

# csv：写 CSV 文件
import csv

# Path：跨平台路径操作
from pathlib import Path

# collections.Counter：统计标签分布，方便打印核对
from collections import Counter

# collections.defaultdict：构造"一键多值"字典，用于按 (日期, 活动区) 分组下采样
from collections import defaultdict

# random：下采样时的随机选择
import random

from config import (
    # 原始数据
    PARA_FLARE_TXT,
    MAGNETOGRAM_DIR,
    # 输出
    INDEX_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    # 时间划分
    TRAIN_PERIODS,
    TEST_PERIODS,
    VAL_PERIODS,
    USE_VAL_SPLIT,
    # 下采样
    DOWNSAMPLE_NEGATIVE,
    VAL_DOWNSAMPLE_NEGATIVE,
    DOWNSAMPLE_SEED,
    # 常量
    NUM_PARAMS,
)


# ============================================================
# 文件名解析
# ============================================================
#
# 官方文件名格式（见数据集说明）：
#     hmi.sharp_720s.<num>.<YYYYMMDD>_<hhmmss>_TAI.magnetogram.fits.jpg
#
# 例：
#     hmi.sharp_720s.1.20100504_160000_TAI.magnetogram.fits.jpg
#                    ↑ ↑              ↑
#                    │ └─ 观测时刻     └─ TAI 时间标准
#                    └─ 太阳活动区编号（sunspot group number）
#
# 另外注意：
#     para_flare.txt 第 1 列写的是 "...magnetogram.fits"（没有 .jpg 后缀），
#     而实际图片文件是 "...magnetogram.fits.jpg"，所以拼路径时要补 ".jpg"。

# 正则：从文件名里同时抠出 活动区号 和 观测时间
_FILENAME_RE = re.compile(
    r"hmi\.sharp_720s\.(?P<ar>\d+)\.(?P<date>\d{8})_(?P<time>\d{6})_TAI"
)


def parse_filename(fname: str):
    """
    从磁图文件名里解析出活动区编号、日期、时间。

    Parameters
    ----------
    fname : str
        形如 'hmi.sharp_720s.1.20100504_160000_TAI.magnetogram.fits'

    Returns
    -------
    (ar_num, date_str, time_str) : tuple[str, str, str]
        ar_num   : 活动区编号，如 '1'
        date_str : 日期，如 '20100504'
        time_str : 时间，如 '160000'
        解析失败时返回 (None, None, None)
    """
    m = _FILENAME_RE.search(fname)
    if m is None:
        return None, None, None
    return m.group("ar"), m.group("date"), m.group("time")


# ============================================================
# 1. 读原始参数表
# ============================================================


def read_para_flare(txt_path: Path = PARA_FLARE_TXT) -> list[dict]:
    """
    读取 para_flare.txt，解析成结构化记录列表。

    Parameters
    ----------
    txt_path : Path
        para_flare.txt 路径。

    Returns
    -------
    records : list[dict]
        每条记录包含：
            fname    : 原始文件名（不含 .jpg）
            ar       : 活动区编号
            date     : 日期 'YYYYMMDD'
            time     : 时间 'hhmmss'
            params   : list[float]，10 个磁场特征参数
            label    : int，0 或 1
            n_missing: 该行参数里 NaN/解析失败的个数（用于统计数据质量）

    说明
    ----
    - 原文件用空格分隔，且行首有前导空格，所以用 split() 自动处理。
    - 实测第 8/9/10 列存在 NaN 与 'nan' 字符串，这里统一转成 float('nan')，
      后续由 dataset.py 里的 SimpleImputer 或 MinMaxScaler 处理。
    - 跳过列数不等于 12 的异常行，并统计数量。
    """
    records = []
    bad_lines = 0

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            # 正常一行 = 1 列文件名 + 10 列参数 + 1 列标签 = 12 列
            if len(parts) != NUM_PARAMS + 2:
                bad_lines += 1
                continue

            fname = parts[0]
            ar, date, time = parse_filename(fname)

            # 解析 10 个参数；'nan' 或空值 → float('nan')
            params = []
            n_missing = 0
            for v in parts[1:-1]:
                try:
                    params.append(float(v))
                except ValueError:
                    params.append(float("nan"))
                    n_missing += 1

            try:
                label = int(float(parts[-1]))
            except ValueError:
                bad_lines += 1
                continue

            records.append(
                {
                    "fname": fname,
                    "ar": ar,
                    "date": date,
                    "time": time,
                    "params": params,
                    "label": label,
                    "n_missing": n_missing,
                }
            )

    print(f"[读取] {txt_path.name}: 有效 {len(records)} 行，跳过异常 {bad_lines} 行")
    return records


# ============================================================
# 2. 按时间段筛选
# ============================================================


def filter_by_periods(records: list[dict], periods: list[tuple[str, str]]) -> list[dict]:
    """
    按时间段筛选记录。

    Parameters
    ----------
    records : list[dict]
        read_para_flare() 的输出。
    periods : list[tuple[str, str]]
        [(起, 止), ...]，闭区间，格式 'YYYYMMDD'。

    Returns
    -------
    kept : list[dict]
        落在任意一个时间段内的记录。
    """
    kept = []
    for r in records:
        if r["date"] is None:
            continue
        for lo, hi in periods:
            if lo <= r["date"] <= hi:
                kept.append(r)
                break
    return kept


# ============================================================
# 3. 负样本下采样
# ============================================================


def downsample_negative(records: list[dict], seed: int = DOWNSAMPLE_SEED) -> list[dict]:
    """
    对负样本（label=0）下采样：同一天、同一活动区的负样本只保留 1 条。

    这直接照搬论文原文的规则：
        "训练集在未发生太阳耀斑每天的相同活动区样本中只选择一条记录，
         使其数量与发生太阳耀斑的样本量接近。"

    为什么这样下采样是合理的
    ------------------------
    同一个活动区在同一天的连续磁图（每 96 分钟一张）高度相似，
    全部保留等于给模型灌大量重复样本，会让负样本"虚多"。
    按 (日期, 活动区) 去重，既保留了"哪些活动区在哪一天是平静的"这个信息，
    又去掉了时间上的冗余。

    正样本【不】下采样（论文里正样本 2988 条是全取的）。

    Parameters
    ----------
    records : list[dict]
        待下采样的记录（应只包含训练集或验证集）。
    seed : int
        随机种子；同一天同一活动区内随机挑哪一条用它控制。

    Returns
    -------
    kept : list[dict]
        正样本全保留 + 下采样后的负样本。
    """
    rng = random.Random(seed)

    positives = [r for r in records if r["label"] == 1]
    negatives = [r for r in records if r["label"] == 0]

    # 按 (日期, 活动区) 分组
    groups = defaultdict(list)
    for r in negatives:
        groups[(r["date"], r["ar"])].append(r)

    kept_neg = []
    for key, items in groups.items():
        # 每组只留一条
        kept_neg.append(rng.choice(items))

    print(
        f"[下采样] 负样本 {len(negatives)} → {len(kept_neg)} "
        f"（按 日期×活动区 共 {len(groups)} 组，每组留 1 条）；"
        f"正样本 {len(positives)} 条全保留"
    )
    return positives + kept_neg


# ============================================================
# 4. 文件名 → 图片路径
# ============================================================


def attach_image_path(records: list[dict], img_dir: Path = MAGNETOGRAM_DIR) -> list[dict]:
    """
    把记录里的文件名映射成真实图片的绝对路径。

    para_flare.txt 第 1 列是 '...magnetogram.fits'，实际文件是 '...magnetogram.fits.jpg'，
    所以要补 '.jpg' 后缀。

    同时做一次存在性检查：找不到的图片会被剔除并统计，避免训练时报 FileNotFoundError。

    Parameters
    ----------
    records : list[dict]
    img_dir : Path
        磁图所在目录。

    Returns
    -------
    kept : list[dict]
        每条记录新增 'img_path' 字段（str，绝对路径）。
    """
    kept = []
    missing = 0
    for r in records:
        # 补 .jpg 后缀
        candidate = f"{r['fname']}.jpg"
        full = img_dir / candidate
        if full.exists():
            r["img_path"] = str(full)
            kept.append(r)
        else:
            missing += 1

    print(f"[路径] 图片存在 {len(kept)} 条，缺失 {missing} 条")
    return kept


# ============================================================
# 5. 写出 CSV
# ============================================================


def write_split_csv(records: list[dict], out_csv: Path) -> None:
    """
    把记录写成 CSV。

    CSV 列结构（13 列，无表头）：
        第 1 列   : 图片绝对路径
        第 2–11 列: 10 个磁场特征参数
        第 12 列  : 标签（0/1）

    Parameters
    ----------
    records : list[dict]
    out_csv : Path
        输出路径。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in records:
            row = [r["img_path"]] + [f"{v:.10g}" for v in r["params"]] + [r["label"]]
            writer.writerow(row)

    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = len(records) - n_pos
    print(f"[写出] {out_csv.name}: 共 {len(records)} 条 (正 {n_pos} / 负 {n_neg})")


# ============================================================
# 6. 阶段入口
# ============================================================


def run_preprocess_stage() -> dict:
    """
    数据构建阶段的完整入口。

    Returns
    -------
    summary : dict
        各 split 的样本数，便于上层打印或记录。
    """
    print("=" * 70)
    print("阶段一：从 SOLAR-STORM2 原始数据构建索引 CSV")
    print("=" * 70)

    # --- 1. 读原始参数表 ---
    all_records = read_para_flare()
    print(f"       标签分布: {dict(Counter(r['label'] for r in all_records))}")

    # --- 2. 按时间划分出「训练池」和「测试集」 ---
    train_pool = filter_by_periods(all_records, TRAIN_PERIODS)
    test_recs = filter_by_periods(all_records, TEST_PERIODS)

    print()
    print(f"[划分] 训练时间段 {TRAIN_PERIODS} → {len(train_pool)} 条")
    print(f"[划分] 测试时间段 {TEST_PERIODS} → {len(test_recs)} 条")

    # --- 3. 从训练池里再挖验证集（论文没提验证集，这是我们自己加的）---
    if USE_VAL_SPLIT:
        # 先取出验证段，剩下的才是真正的训练集
        val_recs = filter_by_periods(train_pool, VAL_PERIODS)
        # 训练集 = 训练池 - 验证段
        val_ids = {id(r) for r in val_recs}          # 用 id 做集合运算，避免内容重复导致的误删
        train_recs = [r for r in train_pool if id(r) not in val_ids]
        print(f"[划分] 从训练池再切出验证段 {VAL_PERIODS} → {len(val_recs)} 条")
    else:
        train_recs, val_recs = train_pool, []

    print(f"[划分] 最终 训练 {len(train_recs)} / 验证 {len(val_recs)} / 测试 {len(test_recs)}")

    # --- 4. 负样本下采样（只对训练集和验证集；测试集按论文用原始数据）---
    print()
    if DOWNSAMPLE_NEGATIVE:
        train_recs = downsample_negative(train_recs, seed=DOWNSAMPLE_SEED)
    else:
        print("[下采样] 已关闭，训练集保留全部负样本")

    if USE_VAL_SPLIT and VAL_DOWNSAMPLE_NEGATIVE and val_recs:
        val_recs = downsample_negative(val_recs, seed=DOWNSAMPLE_SEED)
    print("[下采样] 测试集不下采样（论文：测试集采用原始数据）")

    # --- 5. 文件名 → 图片路径 ---
    print()
    train_recs = attach_image_path(train_recs)
    val_recs = attach_image_path(val_recs)
    test_recs = attach_image_path(test_recs)

    # --- 6. 写出三份 CSV ---
    print()
    write_split_csv(train_recs, TRAIN_CSV)
    if USE_VAL_SPLIT:
        write_split_csv(val_recs, VAL_CSV)
    write_split_csv(test_recs, TEST_CSV)

    summary = {
        "train": len(train_recs),
        "val": len(val_recs),
        "test": len(test_recs),
    }

    print()
    print("=" * 70)
    print("阶段一完成")
    print("=" * 70)
    print(f"  训练集: {summary['train']} 条 → {TRAIN_CSV}")
    if USE_VAL_SPLIT:
        print(f"  验证集: {summary['val']} 条 → {VAL_CSV}")
    print(f"  测试集: {summary['test']} 条 → {TEST_CSV}")
    print()
    print("论文对照（论文 4.1 节）：")
    print("  训练集 正 2988 / 负 5073")
    print("  测试集 正 196  / 负 11852（论文数字疑有误，见 config.py 说明）")

    return summary


# ============================================================
# 7. 简单自测
# ============================================================

if __name__ == "__main__":
    run_preprocess_stage()
