"""
======================================================================
scripts/run_multiseed.py
一键跑多种子实验：train → test（验证集选阈值）→ 汇总 mean ± std
======================================================================

为什么需要这个脚本
------------------
5 个种子 × (训练 + 测试) 手动要敲 10 条命令，而且任何一条参数写错（漏了 --amp、
忘了 --val-threshold）都会让结果不可比。这个脚本把流程固定下来。

它做的事（对每个种子）：
    1. python main.py train --tag <tag> --seed <s> [--amp] --epochs <N>
    2. python main.py test  --tag <tag> --seed <s> --val-threshold
    3. 跳过已完成的部分（断点续跑：已有 checkpoint 就不再训练）
最后：
    4. python scripts/aggregate_seeds.py --pattern "<tag>_seed*"

用法
----
    # 推荐：3 个种子先看趋势（约 4.5 小时）
    python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 --epochs 8 --amp

    # 一次到位 5 个种子（约 7.5 小时，可过夜）
    python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 4 5 --epochs 8 --amp

    # 只跑测试（训练已完成，想重新评估）
    python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 --test-only

    # 不带 AMP（更慢，但最保守）
    python scripts/run_multiseed.py --tag jwfd --seeds 1 2 3 --epochs 8

注意
----
- 每个种子的产物名：<tag>_seed<s>_best.pth / <tag>_seed<s>_history.csv / *_<tag>_seed<s>_metrics.json
- 脚本会实时打印每个步骤的命令和退出码；某个种子失败不会中断其余的（会记录并在最后汇总）
- 所有输出同时追加写入 experiments/logs/multiseed_<tag>.log
======================================================================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def log(msg: str, logfile: Path | None = None):
    """同时打印到屏幕和日志文件。"""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list[str], logfile: Path | None) -> int:
    """
    执行一条命令，实时透传输出，返回退出码。

    ⚠️ 输出缓冲的坑（2026-10-17 踩过一次）
    -------------------------------------
    一开始只写了 subprocess.PIPE + bufsize=1，以为能看到实时输出，结果
    跑 20 分钟一个 epoch 都没打印 —— 因为 **Python 的 stdout 在"连接到管道"时
    默认是块缓冲（8 KB），不是行缓冲**。子进程每轮那行才 150 字节，
    要积累到 8 KB 或进程退出才吐出来。

    修复：给子进程设置环境变量 PYTHONUNBUFFERED=1（等价于 python -u），
    强制它不缓冲，这样 print() 立刻写到管道，我们才能实时转发。
    """
    log("  执行: " + " ".join(cmd), logfile)
    t0 = time.time()

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"      # ★ 关键：禁止子进程缓冲 stdout
    env["PYTHONIOENCODING"] = "utf-8"  # ★ 避免 Windows 下中文乱码

    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # 过滤掉刷屏的警告
        s = line.rstrip()
        if any(k in s for k in ("UserWarning", "warnings.warn", "Glyph", "findfont")):
            continue
        print(s, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(s + "\n")
    rc = proc.wait()
    log(f"  完成，退出码 {rc}，耗时 {(time.time()-t0)/60:.1f} 分钟", logfile)
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="一键多种子实验（train → test → 汇总）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--tag", type=str, default="jwfd", help="实验标签前缀（默认 jwfd）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="种子列表（默认 1 2 3）")
    ap.add_argument("--epochs", type=int, default=8, help="每个种子的训练轮数（默认 8）")
    ap.add_argument("--amp", action="store_true", help="启用 bf16 混合精度（约 1.78×）")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    ap.add_argument("--test-only", action="store_true", help="跳过训练，只跑测试")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：已有 checkpoint 的种子跳过训练")
    ap.add_argument("--no-aggregate", action="store_true", help="不自动跑汇总")
    args = ap.parse_args()

    logfile = config.LOGS_DIR / f"multiseed_{args.tag}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    # ★ 关键：用 utf-8-sig（带 BOM）创建日志文件。
    # Windows 的 PowerShell 5.1 的 Get-Content 在【没有 BOM】时会按系统 ANSI(cp936/GBK)
    # 解码，导致 UTF-8 中文显示成 "澶氱瀛愬疄楠" 这类乱码。
    # 加上 BOM 之后，PowerShell / 记事本 / VS Code 都能自动识别为 UTF-8。
    # 注意：BOM 只在文件开头写一次，后续 log() 用普通 utf-8 追加即可。
    logfile.write_text("", encoding="utf-8-sig")

    log("=" * 74, logfile)
    log(f"多种子实验开始 | tag={args.tag} | seeds={args.seeds}", logfile)
    log(f"epochs={args.epochs} | amp={args.amp} | test_only={args.test_only}", logfile)
    log(f"日志: {logfile}", logfile)
    log("=" * 74, logfile)

    total_t0 = time.time()
    results: dict[int, dict] = {}

    for seed in args.seeds:
        ckpt = config.CHECKPOINTS_DIR / f"{args.tag}_seed{seed}_best.pth"
        log("", logfile)
        log(f"---------- seed {seed} ----------", logfile)

        # ---------- 1. 训练 ----------
        need_train = (not args.test_only) and not (args.resume and ckpt.exists())
        if need_train:
            train_cmd = [sys.executable, "main.py", "train",
                         "--tag", args.tag, "--seed", str(seed),
                         "--epochs", str(args.epochs)]
            if args.amp:
                train_cmd.append("--amp")
            if args.batch_size is not None:
                train_cmd += ["--batch-size", str(args.batch_size)]
            rc = run(train_cmd, logfile)
            results.setdefault(seed, {})["train_rc"] = rc
            if rc != 0:
                log(f"  [!] seed {seed} 训练失败（rc={rc}），跳过该种子的测试", logfile)
                continue
        else:
            if ckpt.exists():
                log(f"  跳过训练（已存在 {ckpt.name}）", logfile)
            else:
                log(f"  [!] 跳过训练但找不到 checkpoint: {ckpt.name}", logfile)
                results.setdefault(seed, {})["train_rc"] = None
                continue
            results.setdefault(seed, {})["train_rc"] = 0

        # ---------- 2. 测试（验证集选阈值）----------
        test_cmd = [sys.executable, "main.py", "test",
                    "--tag", args.tag, "--seed", str(seed), "--val-threshold"]
        rc = run(test_cmd, logfile)
        results.setdefault(seed, {})["test_rc"] = rc

    # ---------- 3. 汇总 ----------
    elapsed = (time.time() - total_t0) / 60
    log("", logfile)
    log("=" * 74, logfile)
    log(f"全部种子处理完成，总耗时 {elapsed/60:.2f} 小时", logfile)
    ok = [s for s, r in results.items() if r.get("test_rc") == 0]
    bad = [s for s, r in results.items() if r.get("test_rc") != 0]
    log(f"成功: {ok}", logfile)
    if bad:
        log(f"失败/跳过: {bad}", logfile)

    if not args.no_aggregate and ok:
        log("", logfile)
        log("---------- 汇总 mean ± std ----------", logfile)
        agg = [sys.executable, "scripts/aggregate_seeds.py",
               "--pattern", f"{args.tag}_seed*",
               "--out", str(config.REPORTS_DIR / f"{args.tag}_seed_summary.md")]
        run(agg, logfile)

    log("", logfile)
    log("接下来可以：", logfile)
    log(f"  experiments/reports/{args.tag}_seed_summary.md   ← 汇总表", logfile)
    log(f"  experiments/logs/multiseed_{args.tag}.log        ← 全部日志", logfile)
    log("=" * 74, logfile)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
