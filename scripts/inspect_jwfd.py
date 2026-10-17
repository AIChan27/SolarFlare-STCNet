"""
======================================================================
inspect_jwfd.py  ——  JW-FD 数据集探测脚本（只读，不改动任何数据）
======================================================================

用途
    把一个 500+ GB 的 JW-FD 目录"体检"清楚，输出一份结构化报告，
    用于决定：哪些必须留、哪些可以删、训练该怎么接入。

特点
    - 纯 Python 标准库 + 可选的 Pillow / pandas / numpy（没装也能跑）
    - **只读**：不修改、不移动、不删除任何文件
    - 大目录体积统计会**抽样估算**，不会遍历全部 50 万张图（避免跑几小时）
    - 结果同时打印到屏幕并写入报告文件

用法
    # 最简（显式指定数据集根目录，推荐）
    python scripts/inspect_jwfd.py --root "C:\\Personal Files\\baidunetdiskdownload\\JW-FD_subset_10pct\\JW-FD_subset_10pct.tar\\JW-FD_subset_10pct"

    # 不指定 root 时脚本会自动搜索常见位置（可能失败，失败会提示）
    python scripts/inspect_jwfd.py

    # 体积要精确统计（慢，会遍历全部文件）
    python scripts/inspect_jwfd.py --root "..." --full-size

    # 指定报告输出位置
    python scripts/inspect_jwfd.py --root "..." --out "D:\\jwfd_report.txt"

运行完把 jwfd_report.txt 的内容贴回给分析方即可。
======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------- 可选依赖（没装也能跑）----------
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ======================================================================
# 输出工具：同时写屏幕和文件
# ======================================================================

class Reporter:
    """收集所有报告行，最后统一打印 + 落盘。"""

    def __init__(self, out_path: Path):
        self.lines: list[str] = []
        self.out_path = out_path

    def __call__(self, text: str = ""):
        self.lines.append(text)
        print(text, flush=True)

    def rule(self, ch: str = "=", n: int = 78):
        self(ch * n)

    def title(self, text: str):
        self()
        self.rule("=")
        self(f"  {text}")
        self.rule("=")

    def section(self, text: str):
        self()
        self.rule("-")
        self(f"  {text}")
        self.rule("-")

    def save(self):
        try:
            self.out_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
            print(f"\n[已保存] 报告写入: {self.out_path}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"\n[警告] 报告保存失败: {e}", flush=True)


def human(n_bytes: float) -> str:
    """把字节数转成人看的单位。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.2f} PB"


# ======================================================================
# 1. 定位数据集根目录
# ======================================================================

MARKERS = ["split_v2_ar_membership.json", "flare_labels.csv"]


def looks_like_root(p: Path) -> int:
    """给一个目录打分，分越高越像数据集根目录。"""
    if not p.is_dir():
        return -1
    try:
        names = {e.name for e in os.scandir(p)}
    except OSError:
        return -1

    score = 0
    for m in MARKERS:
        if m in names:
            score += 10
    if any(n.startswith("png_600_Th") for n in names):
        score += 10
    if "fits_600" in names:
        score += 8
    if any(n.startswith("solar_flare_dataset") for n in names):
        score += 8
    if any(n.startswith("movies") for n in names):
        score += 3
    if "jw-fd" in p.name.lower():
        score += 2
    return score


def find_root(hint: Path | None) -> Path | None:
    """定位数据集根目录。用户显式指定则直接尊重。"""
    if hint is not None:
        return hint

    candidates_starts = [
        Path.cwd(),
        Path.home() / "Desktop",
        Path("C:/Personal Files/baidunetdiskdownload"),
        Path("D:/"),
        Path("E:/"),
    ]

    best: tuple[int, Path] | None = None
    for start in candidates_starts:
        if not start.exists():
            continue
        try:
            for dirpath, dirnames, _ in os.walk(start):
                depth = len(Path(dirpath).relative_to(start).parts)
                if depth > 4:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d.lower() not in
                               ("windows", "$recycle.bin", "system volume information",
                                "node_modules", ".git", "appdata")]
                score = looks_like_root(Path(dirpath))
                if score > 0 and (best is None or score > best[0]):
                    best = (score, Path(dirpath))
        except (OSError, PermissionError, ValueError):
            continue

    return best[1] if best else None


# ======================================================================
# 2. 目录体积统计（大目录抽样估算）
# ======================================================================

def dir_stats(d: Path, full: bool, sample_cap: int = 3000) -> dict:
    """
    统计一个目录下的文件数、总体积、扩展名分布。

    为避免遍历 50 万张图花掉几小时：
      - 文件数：完整计数（os.scandir 只读目录项，很快）
      - 总体积：超过 sample_cap 个文件时**抽样估算**（除非 full=True）
    """
    n_files = 0
    ext_counter: Counter = Counter()
    sizes: list[int] = []
    first_files: list[Path] = []
    truncated = False
    t0 = time.time()

    for dirpath, _dirnames, filenames in os.walk(d):
        for fn in filenames:
            n_files += 1
            ext_counter[os.path.splitext(fn)[1].lower()] += 1
            if len(first_files) < 3:
                first_files.append(Path(dirpath) / fn)
            if full or len(sizes) < sample_cap:
                try:
                    sizes.append(os.path.getsize(os.path.join(dirpath, fn)))
                except OSError:
                    pass
            if not full and n_files >= 500_000:
                truncated = True
                break
        if truncated:
            break

    total = sum(sizes)
    if sizes and not full and n_files > len(sizes):
        est_total = total / len(sizes) * n_files
        estimated = True
    else:
        est_total = total
        estimated = False

    return {
        "n_files": n_files,
        "total_bytes": est_total,
        "estimated": estimated,
        "n_sampled": len(sizes),
        "exts": ext_counter.most_common(6),
        "first_files": first_files,
        "truncated": truncated,
        "seconds": time.time() - t0,
    }


# ======================================================================
# 3. PNG 分支：档位、数量、真实图片格式
# ======================================================================

def inspect_png_branches(root: Path, rep: Reporter, sample_n: int = 12) -> dict:
    rep.section("3. PNG 分支（十档饱和阈值）总览")

    def th_of(p: Path) -> int:
        m = re.search(r"Th(\d+)", p.name)
        return int(m.group(1)) if m else 0

    branches = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("png_600_Th")],
        key=th_of,
    )

    if not branches:
        rep("  [未找到 png_600_Th* 目录]")
        return {}

    rep(f"  共发现 {len(branches)} 档 PNG 分支")
    rep()
    rep(f"  {'分支目录':<24} {'PNG 张数':>12} {'AR 目录数':>12}")
    rep("  " + "-" * 50)

    info: dict[str, dict] = {}
    for b in branches:
        n_png = 0
        n_subdirs = 0
        try:
            subs = [e for e in os.scandir(b) if e.is_dir()]
            n_subdirs = len(subs)
            for s in subs:
                for _dp, _dn, fns in os.walk(s.path):
                    n_png += sum(1 for f in fns if f.lower().endswith(".png"))
        except OSError as e:
            rep(f"  {b.name:<24} [读取失败: {e}]")
            continue
        info[b.name] = {"n_png": n_png, "n_ar": n_subdirs}
        rep(f"  {b.name:<24} {n_png:>12,} {n_subdirs:>12,}")

    # ---------- 真实图片格式抽查 ----------
    rep()
    rep("  [图片格式抽查]")
    if not HAS_PIL:
        rep("    未安装 Pillow，跳过格式检查（可 pip install Pillow）")
        return info

    b0 = branches[0]
    all_pngs: list[Path] = []
    try:
        for s in os.scandir(b0):
            if s.is_dir():
                for dp, _dn, fns in os.walk(s.path):
                    for f in fns:
                        if f.lower().endswith(".png"):
                            all_pngs.append(Path(dp) / f)
                            if len(all_pngs) >= 200:
                                break
                    if len(all_pngs) >= 200:
                        break
            if len(all_pngs) >= 200:
                break
    except OSError:
        pass

    if not all_pngs:
        rep(f"    {b0.name} 下没扫到 PNG")
        return info

    picks = random.sample(all_pngs, min(sample_n, len(all_pngs)))
    modes: Counter = Counter()
    sizes_: Counter = Counter()
    fsz: list[int] = []
    for p in picks:
        try:
            fsz.append(p.stat().st_size)
            with Image.open(p) as im:
                modes[im.mode] += 1
                sizes_[im.size] += 1
        except Exception as e:                                    # noqa: BLE001
            rep(f"    打开失败 {p.name}: {e}")

    rep(f"    分支        : {b0.name}")
    rep(f"    抽样张数    : {len(picks)}")
    rep(f"    色彩模式    : {dict(modes)}   （L = 单通道灰度）")
    rep(f"    尺寸分布    : {dict(sizes_)}")
    if fsz:
        avg = sum(fsz) / len(fsz)
        rep(f"    单张体积    : 平均 {human(avg)}，最小 {human(min(fsz))}，"
            f"最大 {human(max(fsz))}")
        n_this = info.get(b0.name, {}).get("n_png", 0)
        if n_this:
            rep(f"    → 按此估算 {b0.name} 总 PNG 体积约 {human(avg * n_this)}")
    rep(f"    示例文件名  : {picks[0].name}")

    # 灰度值分布（验证 0 G = 127）
    if HAS_NUMPY:
        try:
            import numpy as np
            with Image.open(picks[0]) as im:
                arr = np.asarray(im)
            rep(f"    [灰度值] min={arr.min()} max={arr.max()} "
                f"mean={arr.mean():.2f} 中位数={int(np.median(arr))}")
            hist = np.bincount(arr.ravel(), minlength=256)
            rep(f"    [灰度值] 灰阶 127 的像素占比 = {hist[127] / arr.size * 100:.2f}%"
                f"   （论文称 0 G 落在灰阶 127）")
        except Exception as e:                                    # noqa: BLE001
            rep(f"    灰度统计失败: {e}")

    return info


# ======================================================================
# 4. CSV：表头、行数、标签分布
# ======================================================================

def inspect_csvs(root: Path, rep: Reporter):
    rep.section("4. CSV 文件（特征 + 标签）")

    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        rep("  [根目录下没有 CSV]")
        return None

    rep(f"  共发现 {len(csvs)} 个 CSV：")
    for c in csvs:
        rep(f"    {c.name:<58} {human(c.stat().st_size)}")

    target = [c for c in csvs if "png" in c.name.lower()] or csvs
    pref = [c for c in target if "Th800" in c.name]
    main_csv = pref[0] if pref else target[0]

    rep()
    rep(f"  【重点分析】{main_csv.name}")

    # ---- 读表头 ----
    try:
        with open(main_csv, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline().rstrip("\n").rstrip("\r")
        cols = [c.strip().strip('"') for c in header_line.split(",")]
        rep(f"  列数: {len(cols)}   （论文称 66 列）")

        feat_like, label_bin, label_cls, meta = [], [], [], []
        for c in cols:
            lc = c.lower()
            if lc.startswith("flare_label"):
                label_bin.append(c)
            elif lc.startswith("flare_class"):
                label_cls.append(c)
            elif lc in ("image_filename", "image_path", "filename", "path"):
                meta.append(c)
            else:
                feat_like.append(c)

        rep()
        rep(f"  特征类列   : {len(feat_like)} 列")
        rep(f"  二分类标签 : {len(label_bin)} 列")
        rep(f"  类别标签   : {len(label_cls)} 列")
        rep(f"  元数据列   : {len(meta)} 列  {meta}")
        rep()
        rep("  [全部列名]")
        for i, c in enumerate(cols, 1):
            tag = ("FEAT" if c in feat_like else
                   "LABEL" if c in label_bin else
                   "CLASS" if c in label_cls else "META")
            rep(f"    {i:>3}. [{tag:<5}] {c}")

        if label_bin:
            rep()
            rep(f"  [二分类标签列清单]（{len(label_bin)} 个）")
            for c in label_bin:
                rep(f"    {c}")
    except Exception as e:                                        # noqa: BLE001
        rep(f"  读表头失败: {e}")
        return None

    # ---- 行数 ----
    n_rows = None
    rep()
    try:
        with open(main_csv, "r", encoding="utf-8", errors="replace") as f:
            n_rows = sum(1 for _ in f) - 1
        rep(f"  数据行数（= 磁图样本数）: {n_rows:,}")
    except Exception as e:                                        # noqa: BLE001
        rep(f"  行数统计失败: {e}")

    # ---- 标签分布 + 特征统计 ----
    if HAS_PANDAS:
        rep()
        rep("  [用 pandas 分析标签分布与特征统计]")
        try:
            df = pd.read_csv(main_csv)
            rep(f"    DataFrame 形状: {df.shape}")

            if label_bin:
                rep()
                rep(f"    {'标签列':<32} {'正样本数':>12} {'正样本占比':>13}")
                rep("    " + "-" * 60)
                for c in label_bin:
                    if c in df.columns:
                        s = pd.to_numeric(df[c], errors="coerce")
                        npos = int((s == 1).sum())
                        rate = npos / max(len(s), 1)
                        rep(f"    {c:<32} {npos:>12,} {rate:>12.4%}")

            if label_cls:
                rep()
                rep("    [类别标签取值分布]（前 3 列）")
                for c in label_cls[:3]:
                    if c in df.columns:
                        vc = df[c].astype(str).value_counts().head(10)
                        rep(f"      {c}: {dict(vc)}")

            if feat_like:
                rep()
                rep("    [29 个特征的统计]")
                rep(f"    {'特征':<34} {'非空':>9} {'缺失':>8} {'min':>12} {'max':>12} {'均值':>12}")
                rep("    " + "-" * 94)
                for c in feat_like:
                    if c not in df.columns:
                        continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    rep(f"    {c[:34]:<34} {s.notna().sum():>9,} {s.isna().sum():>8,} "
                        f"{s.min():>12.4g} {s.max():>12.4g} {s.mean():>12.4g}")

            rep()
            rep("    [image_path / image_filename 样例]")
            for c in meta:
                if c in df.columns:
                    for v in df[c].head(3):
                        rep(f"      {c} = {v}")
        except Exception as e:                                    # noqa: BLE001
            rep(f"    pandas 分析失败: {e}")
    else:
        rep()
        rep("  未安装 pandas，跳过标签分布分析（可 pip install pandas）")

    fits_csvs = [c for c in csvs if "fits" in c.name.lower()]
    if fits_csvs:
        rep()
        rep(f"  [FITS 分支 CSV] {[c.name for c in fits_csvs]}")
        rep("    （与 PNG 分支列结构应一致，只是图像来源不同）")

    return n_rows


# ======================================================================
# 5. 官方划分名单
# ======================================================================

def inspect_split(root: Path, rep: Reporter):
    rep.section("5. 官方划分名单 split_v2_ar_membership.json")

    p = root / "split_v2_ar_membership.json"
    if not p.exists():
        found = list(root.rglob("split_v2_ar_membership.json"))
        if not found:
            rep("  [未找到 split_v2_ar_membership.json]")
            return
        p = found[0]
        rep(f"  （在子目录找到）{p}")

    rep(f"  文件: {p}")
    rep(f"  体积: {human(p.stat().st_size)}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                        # noqa: BLE001
        rep(f"  解析失败: {e}")
        return

    rep(f"  顶层类型: {type(data).__name__}")
    if isinstance(data, dict):
        rep(f"  顶层键  : {list(data.keys())}")
        rep()
        total = 0
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                rep(f"    {k:<16} {len(v):>6,} 个活动区   "
                    f"示例: {list(v[:5]) if v else '[]'}")
                total += len(v)
            elif isinstance(v, dict):
                rep(f"    {k:<16} dict，{len(v)} 个键  示例: {list(v.keys())[:5]}")
            else:
                rep(f"    {k:<16} {v}")
        if total:
            rep()
            rep(f"  三个分区活动区总数: {total:,}")
    elif isinstance(data, list):
        rep(f"  列表长度: {len(data)}  示例: {data[:5]}")


# ======================================================================
# 6. 交叉核对：CSV 里的图片是否真的存在
# ======================================================================

def inspect_consistency(root: Path, rep: Reporter, max_check: int = 300):
    rep.section("6. 交叉核对（CSV ↔ PNG 是否对得上）")

    if not HAS_PANDAS:
        rep("  需要 pandas，跳过")
        return

    csvs = [c for c in root.glob("*.csv") if "png" in c.name.lower()]
    if not csvs:
        rep("  没找到 PNG 分支 CSV，跳过")
        return
    pref = [c for c in csvs if "Th800" in c.name]
    main_csv = pref[0] if pref else csvs[0]

    try:
        df = pd.read_csv(main_csv, nrows=5000)
    except Exception as e:                                        # noqa: BLE001
        rep(f"  读 CSV 失败: {e}")
        return

    path_col = next((c for c in df.columns if "path" in c.lower()), None)
    fn_col = next((c for c in df.columns if "filename" in c.lower()), None)

    rep(f"  CSV: {main_csv.name}")
    rep(f"  路径列: {path_col}   文件名列: {fn_col}")

    if path_col is None and fn_col is None:
        rep("  找不到图片路径/文件名列，无法核对")
        return

    n = min(max_check, len(df))
    sample = df.sample(n=n, random_state=0) if len(df) > n else df

    ok, miss = 0, 0
    miss_examples: list[str] = []
    for _, row in sample.iterrows():
        cand: list[Path] = []
        if path_col and isinstance(row[path_col], str):
            raw = row[path_col].replace("\\", "/")
            cand.append(Path(raw))
            cand.append(root / raw.lstrip("/"))
        if fn_col and isinstance(row[fn_col], str):
            fn = row[fn_col]
            cand.append(root / "png_600_Th800" / fn)
            cand.append(root / fn)

        if any(c.exists() for c in cand):
            ok += 1
        else:
            miss += 1
            if len(miss_examples) < 3:
                miss_examples.append(str(cand[0]) if cand else "")

    rep(f"  抽查 {n} 行：找到图片 {ok}，未找到 {miss}")
    if miss_examples:
        rep("  未找到示例（说明路径列可能是相对路径，需拼接根目录）:")
        for e in miss_examples:
            rep(f"    {e}")
    rep("  ✅ CSV 与图片能对上" if ok > 0 else
        "  ⚠️ 全部未找到——路径列可能是相对路径，需按实际结构调整拼接逻辑")


# ======================================================================
# 7. 体积预算与清理建议
# ======================================================================

def advice(root: Path, rep: Reporter, top_stats: dict, png_info: dict, n_rows):
    rep.section("7. 体积预算与清理建议（依据以上实测）")

    grand = sum(v["total_bytes"] for v in top_stats.values())
    rep(f"  数据集总体积（顶层条目合计）: {human(grand)}")
    rep()
    rep(f"  {'顶层条目':<28} {'文件数':>12} {'体积':>14} {'占比':>8}")
    rep("  " + "-" * 68)
    for name, st in sorted(top_stats.items(), key=lambda x: -x[1]["total_bytes"]):
        pct = st["total_bytes"] / grand * 100 if grand else 0
        flag = " (抽样)" if st["estimated"] else ""
        rep(f"  {name[:28]:<28} {st['n_files']:>12,} "
            f"{human(st['total_bytes']):>14} {pct:>7.1f}%{flag}")

    rep()
    rep("  [我们的训练需要什么]")
    rep("    ✅ 需要: 1 档 png_600_Th*（推荐 Th800）+ 对应 CSV")
    rep("             + split_v2_ar_membership.json + flare_labels.csv")
    rep("    ❌ 不需要: fits_600/、其余 9 档 png_600_Th*、movies_*/")
    rep()

    keep_bytes = 0
    drop_names = []
    for name, st in top_stats.items():
        nl = name.lower()
        keep = (nl == "png_600_th800" or nl.endswith(".csv") or nl.endswith(".json"))
        if keep:
            keep_bytes += st["total_bytes"]
        else:
            drop_names.append(f"{name} ({human(st['total_bytes'])})")

    rep(f"  预计【保留】体积: {human(keep_bytes)}")
    if drop_names:
        rep(f"  预计【可删】条目: {len(drop_names)} 个")
        for d in drop_names:
            rep(f"      - {d}")
        rep()
        rep("  ⚠️ 重要提醒：'可删'是**接入训练之后**的判断。")
        rep("     请先确认第 3 节里 Th800 分支确实存在且完整，再决定删除。")
        rep("     删除前建议先确认能跑通训练，再把不要的移出 SSD。")

    rep()
    rep("  [离线预缩到 224×224 的收益估算]")
    th800 = png_info.get("png_600_Th800", {})
    n_png = th800.get("n_png") or (n_rows or 0)
    if n_png:
        rep(f"    本档 PNG 张数: {n_png:,}")
        rep(f"    原始 600×600 PNG（按 150 KB/张）→ {human(n_png * 150 * 1024)}")
        for kb in (20, 25, 30):
            rep(f"    若 224×224 JPEG 约 {kb} KB/张      → {human(n_png * kb * 1024)}")
        rep("    >>> 预计可压缩到原来的 1/6 ~ 1/5，且训练时不再需要 Resize")

    rep()
    rep("  [下一步建议]")
    rep("    1. 确认 png_600_Th800 分支完整（见第 3 节）")
    rep("    2. 用 224×224 JPEG 预缩替代原图（见上）")
    rep("    3. 预缩完成后，把 fits_600/ + 其余 9 档 + movies/ 移出 SSD 或删除")
    rep("    4. 把 jwfd_report.txt 贴回给分析方，以便生成接入代码")


# ======================================================================
# main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="JW-FD 数据集探测（只读）")
    ap.add_argument("--root", type=str, default=None, help="数据集根目录")
    ap.add_argument("--full-size", action="store_true", help="体积统计不抽样（慢）")
    ap.add_argument("--sample", type=int, default=3000, help="体积抽样文件数上限")
    ap.add_argument("--out", type=str, default="jwfd_report.txt", help="报告输出路径")
    args = ap.parse_args()

    rep = Reporter(Path(args.out).resolve())

    rep.rule("=")
    rep("  JW-FD 数据集探测报告（只读）")
    rep(f"  生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    rep(f"  Python  : {sys.version.split()[0]}")
    rep(f"  可用依赖: Pillow={HAS_PIL}  pandas={HAS_PANDAS}  numpy={HAS_NUMPY}")
    rep.rule("=")

    # ---------- 1. 定位根目录 ----------
    rep.title("1. 定位数据集根目录")
    hint = Path(args.root) if args.root else None
    root = find_root(hint)
    if root is None or not root.exists():
        rep("  ❌ 没能自动定位数据集根目录。")
        rep("     请用 --root 显式指定，例如：")
        rep(r'     python scripts/inspect_jwfd.py --root "C:\Personal Files\baidunetdiskdownload\JW-FD_subset_10pct\JW-FD_subset_10pct.tar\JW-FD_subset_10pct"')
        rep.save()
        return 1

    rep(f"  数据集根目录: {root}")
    rep(f"  是否目录    : {root.is_dir()}")

    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError as e:
        rep(f"  无法读取目录: {e}")
        rep.save()
        return 1

    rep()
    rep(f"  顶层共 {len(entries)} 个条目：")
    for e in entries:
        kind = "DIR " if e.is_dir() else "FILE"
        try:
            sz = "" if e.is_dir() else f"  {human(e.stat().st_size)}"
        except OSError:
            sz = ""
        rep(f"    [{kind}] {e.name}{sz}")

    # ---------- 2. 各顶层条目体积 ----------
    rep.title("2. 各顶层条目：文件数与体积")
    if not args.full_size:
        rep(f"  （大目录按前 {args.sample} 个文件抽样估算；加 --full-size 可精确统计）")
    top_stats: dict[str, dict] = {}
    for e in entries:
        if not e.is_dir():
            try:
                top_stats[e.name] = {
                    "n_files": 1, "total_bytes": e.stat().st_size,
                    "estimated": False, "n_sampled": 1,
                    "exts": [(os.path.splitext(e.name)[1].lower(), 1)],
                    "first_files": [Path(e.path)], "truncated": False, "seconds": 0.0,
                }
            except OSError:
                pass
            continue

        rep()
        rep(f"  正在统计: {e.name} ...")
        st = dir_stats(Path(e.path), args.full_size, args.sample)
        top_stats[e.name] = st
        ext_str = ", ".join(f"{k or '(无扩展名)'}×{v:,}" for k, v in st["exts"])
        rep(f"    文件数 : {st['n_files']:,}" + ("  (统计被截断)" if st["truncated"] else ""))
        rep(f"    体积   : {human(st['total_bytes'])}"
            + (f"  ← 抽样 {st['n_sampled']:,} 个文件估算" if st["estimated"] else "  (精确)"))
        rep(f"    扩展名 : {ext_str}")
        rep(f"    耗时   : {st['seconds']:.1f}s")

    # ---------- 3~7 ----------
    rep.title("3. PNG 分支详情")
    png_info = inspect_png_branches(root, rep, sample_n=12)

    rep.title("4. CSV 详情")
    n_rows = inspect_csvs(root, rep)

    rep.title("5. 官方划分名单")
    inspect_split(root, rep)

    rep.title("6. CSV ↔ 图片一致性")
    inspect_consistency(root, rep)

    rep.title("7. 体积预算与清理建议")
    advice(root, rep, top_stats, png_info, n_rows)

    rep()
    rep.rule("=")
    rep("  探测完成")
    rep.rule("=")

    rep.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
