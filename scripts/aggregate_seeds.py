"""
----------------------------------------------------------------------
scripts/aggregate_seeds.py
汇总多种子实验的结果：mean ± std + 每个种子的明细

用法（在项目根目录运行）：
    python scripts/aggregate_seeds.py                     # 汇总全部
    python scripts/aggregate_seeds.py --pattern "stcnet_seed*"
    python scripts/aggregate_seeds.py --out experiments/reports/seed_summary.md

输入：experiments/reports/*_metrics.json（由 main.py test 生成）
输出：终端表格 + 可选的 markdown 文件
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORTS_DIR = PROJECT_ROOT / "experiments" / "reports"

# 要汇总的指标
METRICS = ["tss", "tpr", "tnr", "far", "precision", "f1", "accuracy", "hss", "auc"]


def load_runs(pattern: str) -> dict:
    """
    读入所有匹配的 metrics.json，返回 {seed: metrics_dict}。

    seed 从文件名里解析，例如 20261016_010203_stcnet_seed3_metrics.json → 3
    """
    runs = {}
    for f in sorted(REPORTS_DIR.glob("*_metrics.json")):
        if not re.search(pattern.replace("*", ".*"), f.name):
            continue
        m = re.search(r"seed(\d+)", f.name)
        if m is None:
            continue                      # 没有 seed 标记的跳过
        seed = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            runs[seed] = json.load(fh)
    return runs


def main():
    ap = argparse.ArgumentParser(description="多种子实验结果汇总")
    ap.add_argument("--pattern", type=str, default="stcnet_seed*",
                    help="文件名匹配模式（默认 stcnet_seed*）")
    ap.add_argument("--out", type=str, default=None,
                    help="同时输出 markdown 文件到该路径")
    args = ap.parse_args()

    runs = load_runs(args.pattern)
    if not runs:
        print(f"[错误] 在 {REPORTS_DIR} 下没找到匹配 '{args.pattern}' 的 metrics.json")
        print("       请先跑完多个种子：for s in 1 2 3 4 5; do python main.py train --seed $s; "
              "python main.py test --tag stcnet --seed $s; done")
        return 1

    seeds = sorted(runs.keys())
    print("=" * 92)
    print(f"多种子实验汇总 | 匹配模式: {args.pattern} | 共 {len(seeds)} 个种子: {seeds}")
    print("=" * 92)

    # ---------- 明细表 ----------
    header = f"{'seed':>5} | " + " ".join(f"{m:>9}" for m in METRICS)
    print(header)
    print("-" * 92)
    for s in seeds:
        r = runs[s]
        row = f"{s:>5} | " + " ".join(f"{r.get(m, float('nan')):>9.4f}" for m in METRICS)
        print(row)

    # ---------- 统计量 ----------
    print("-" * 92)
    print(f"{'mean':>5} | " + " ".join(
        f"{st.mean(runs[s].get(m, 0) for s in seeds):>9.4f}" for m in METRICS))
    print(f"{'std':>5} | " + " ".join(
        f"{st.stdev([runs[s].get(m, 0) for s in seeds]) if len(seeds) > 1 else 0:>9.4f}"
        for m in METRICS))
    print(f"{'min':>5} | " + " ".join(
        f"{min(runs[s].get(m, 0) for s in seeds):>9.4f}" for m in METRICS))
    print(f"{'max':>5} | " + " ".join(
        f"{max(runs[s].get(m, 0) for s in seeds):>9.4f}" for m in METRICS))
    print("=" * 92)

    # ---------- 重点指标 ----------
    print()
    print("【重点指标】")
    for m in ["tss", "tpr", "far"]:
        vals = [runs[s].get(m, 0) for s in seeds]
        if len(vals) > 1:
            mean, sd = st.mean(vals), st.stdev(vals)
            print(f"  {m.upper():>4}: {mean:.4f} ± {sd:.4f}   "
                  f"(95% 置信区间约 [{mean - 1.96 * sd / len(vals) ** .5:.4f}, "
                  f"{mean + 1.96 * sd / len(vals) ** .5:.4f}])")
        else:
            print(f"  {m.upper():>4}: {vals[0]:.4f}  （只有 1 个种子，无法算 std）")

    print()
    if len(seeds) < 3:
        print("⚠️  建议至少 3 个种子（推荐 5 个）才能得到可靠的 std 估计。")

    # ---------- 可选输出 markdown ----------
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 多种子实验结果汇总", "",
                 f"- 匹配模式：`{args.pattern}`",
                 f"- 种子：{seeds}", f"- 种子数：{len(seeds)}", "",
                 "| " + " | ".join(["seed"] + [m.upper() for m in METRICS]) + " |",
                 "|" + "---|" * (len(METRICS) + 1)]
        for s in seeds:
            r = runs[s]
            lines.append("| " + " | ".join(
                [str(s)] + [f"{r.get(m, float('nan')):.4f}" for m in METRICS]) + " |")
        for stat_name, fn in [("mean", st.mean),
                              ("std", lambda v: st.stdev(v) if len(v) > 1 else 0.0)]:
            lines.append("| **" + stat_name + "** | " + " | ".join(
                f"{fn([runs[s].get(m, 0) for s in seeds]):.4f}" for m in METRICS) + " |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[输出] markdown 汇总已保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())