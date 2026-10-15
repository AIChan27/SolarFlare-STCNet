"""
----------------------------------------------------------------------
src/utils/paths.py
路径工具：统一获取各目录、压缩长路径用于日志输出

两个作用：
    1. 目录工厂函数：需要哪个目录就调哪个，不用在各处手写 Path 拼接
    2. shorten_path  ：把绝对路径转成相对路径，避免在终端/日志里
                      暴露完整的个人目录结构
----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    INDEX_DIR,
    EXPERIMENTS_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PRETRAINED_DIR,
)


# ============================================================
# 1. 目录工厂
# ============================================================


def project_root() -> Path:
    """返回项目根目录。"""
    return PROJECT_ROOT


def data_dir() -> Path:
    """返回数据目录（data/）。"""
    return DATA_DIR


def index_dir() -> Path:
    """返回索引 CSV 所在目录（data/index/）。"""
    return INDEX_DIR


def experiments_dir() -> Path:
    """返回实验产物根目录（experiments/）。"""
    return EXPERIMENTS_DIR


def logs_dir() -> Path:
    """返回日志目录（experiments/logs/）。"""
    return LOGS_DIR


def checkpoints_dir() -> Path:
    """返回模型权重目录（experiments/checkpoints/）。"""
    return CHECKPOINTS_DIR


def figures_dir() -> Path:
    """返回配图目录（experiments/figures/）。"""
    return FIGURES_DIR


def reports_dir() -> Path:
    """返回实验报告目录（experiments/reports/）。"""
    return REPORTS_DIR


def pretrained_dir() -> Path:
    """返回预训练权重目录（pretrained/）。"""
    return PRETRAINED_DIR


def ensure_dir(path: Path) -> Path:
    """
    保证目录存在，不存在就创建。

    Parameters
    ----------
    path : Path
        目标目录。

    Returns
    -------
    path : Path
        原样返回，方便链式调用：ensure_dir(x) / "a.csv"
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ============================================================
# 2. 路径压缩（用于日志）
# ============================================================


def shorten_path(path) -> str:
    """
    把绝对路径转成相对路径，相对于项目根目录的父目录。

    示例:
        C:\\Personal Files\\DeepLearning\\SolarFlare-STCNet\\experiments\\logs\\x.csv
        → SolarFlare-STCNet\\experiments\\logs\\x.csv

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    str
        相对路径；无法转换时原样返回。
    """
    this_file = Path(__file__).resolve()
    # src/utils/paths.py → src/utils → src → 项目根目录
    project_root_dir = this_file.parent.parent.parent
    base = project_root_dir.parent

    try:
        return str(Path(path).resolve().relative_to(base))
    except (ValueError, OSError):
        return str(path)


# ============================================================
# 3. 自测
# ============================================================

if __name__ == "__main__":
    for name, fn in [
        ("项目根", project_root),
        ("数据", data_dir),
        ("索引", index_dir),
        ("日志", logs_dir),
        ("权重", checkpoints_dir),
        ("配图", figures_dir),
        ("预训练", pretrained_dir),
    ]:
        p = fn()
        print(f"{name:6s} {p}")
        print(f"       → {shorten_path(p)}")
