"""
----------------------------------------------------------------------
setup_project.py
目录初始化脚本，clone 后运行一次

作用：
    创建项目运行所需的全部目录，并检查关键文件是否存在。
    这样在全新环境下第一次跑 main.py 之前，先跑这个脚本就能把目录补齐，
    不会因为"目录不存在"报错。

用法：
    python setup_project.py
----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# 需要创建的目录（从 config 里取，避免两处维护）
DIRS_TO_CREATE = [
    ("数据目录", config.DATA_DIR),
    ("索引目录", config.INDEX_DIR),
    ("归一化统计", config.STATS_DIR),
    ("实验产物", config.EXPERIMENTS_DIR),
    ("训练日志", config.LOGS_DIR),
    ("模型权重", config.CHECKPOINTS_DIR),
    ("论文配图", config.FIGURES_DIR),
    ("实验报告", config.REPORTS_DIR),
    ("文档", PROJECT_ROOT / "docs"),
    ("文档-笔记", PROJECT_ROOT / "docs" / "notes"),
    ("文档-排错", PROJECT_ROOT / "docs" / "troubleshooting"),
    ("文档-科研", PROJECT_ROOT / "docs" / "scientific_research"),
    ("Notebook", PROJECT_ROOT / "notebooks"),
    ("独立脚本", PROJECT_ROOT / "scripts"),
    ("预训练权重", config.PRETRAINED_DIR),
]


def main():
    print("=" * 70)
    print("项目目录初始化")
    print("=" * 70)
    print(f"项目根目录: {PROJECT_ROOT}")
    print()

    # ---------- 1. 创建目录 ----------
    print("[1] 创建目录")
    for name, d in DIRS_TO_CREATE:
        d = Path(d)
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        flag = "已存在" if existed else "已创建"
        print(f"    [{flag}] {name:12s} {d}")
    print()

    # ---------- 2. 检查关键文件 ----------
    print("[2] 检查关键文件")
    checks = [
        ("SOLAR-STORM2 磁图目录", config.MAGNETOGRAM_DIR),
        ("SOLAR-STORM2 参数表", config.PARA_FLARE_TXT),
        ("Swin 预训练权重", config.SWIN_PRETRAINED),
        ("CNN 预训练权重", config.CNN_PRETRAINED),
    ]
    all_ok = True
    for name, p in checks:
        ok = Path(p).exists()
        all_ok &= ok
        print(f"    [{'OK ' if ok else '缺失'}] {name:24s} {p}")
    print()

    # ---------- 3. 提示下一步 ----------
    print("=" * 70)
    if all_ok:
        print("初始化完成。下一步：")
        print("    python main.py build-data      # 构建索引 CSV")
        print("    python main.py train --smoke   # 冒烟测试")
        print("    python main.py train           # 正式训练")
    else:
        print("初始化完成，但有关键文件缺失。")
        print("请按上面的路径把 SOLAR-STORM2 数据集和预训练权重放到位。")
        print("数据集下载：https://tianchi.aliyun.com/dataset/74780")
        print("预训练权重：")
        print("  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
              "swin_tiny_patch4_window7_224.pth")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
