"""
======================================================================
scripts/check_env.py
环境自检：确认当前 Python / conda 环境是否装好了本项目需要的包
======================================================================

用途
----
换机器时（比如拿到导师的 Tesla P40 机器上）第一步跑这个脚本，
它会告诉你：**当前这个环境到底缺什么**。

只做"检测"，**不会安装、不会修改任何东西**。

怎么用
------
    # 1) 先激活 conda 环境
    conda activate 你的环境名

    # 2) 再跑这个脚本
    python scripts\\check_env.py

看最后一行的"结论"：
    - 全部就绪   -> 退出码 0
    - 有缺失     -> 退出码 1，并给出安装命令

为什么要有这个脚本
------------------
同一台电脑上常常有多个 Python（系统 Python、conda base、项目 conda 环境），
很容易出现"在 A 环境装了包，却在 B 环境跑程序"的情况。本脚本会打印
**当前正在使用的 Python 解释器绝对路径**，一眼就能看出用的是哪个。

注意事项
--------
1. 本脚本**故意不在文件顶部 import 第三方包**，而是用 importlib 探测，
   这样即使某个包没装，脚本本身也能正常运行并报告"缺哪个"。
2. Windows 中文控制台是 GBK 编码，**不能打印 emoji**（会抛 UnicodeEncodeError），
   所以本文件所有输出都用 ASCII 标记 [OK] / [缺]。
======================================================================
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import platform
import sys
from pathlib import Path

# 项目根目录（本文件在 <项目根>/scripts/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ======================================================================
# 依赖清单
# ----------------------------------------------------------------------
# 每项：(import 名, pip 包名, 最低版本, 用途说明)
# 最低版本是本项目代码实际用到的 API 决定的，见下方注释。
# ======================================================================
REQUIRED = [
    # torch.load(..., weights_only=False) 需要 1.13+；
    # torch.amp / torch.autocast 的用法在更早版本也有兼容分支。
    ("torch", "torch", "1.13", "深度学习框架（模型训练/推理）"),
    # transforms.RandomResizedCrop / Resize 等，配合 torch 的版本
    ("torchvision", "torchvision", "0.14", "图像变换 transforms"),
    ("numpy", "numpy", "1.21", "数值计算、数组操作"),
    ("pandas", "pandas", "1.3", "读取 66 列索引 CSV"),
    ("sklearn", "scikit-learn", "1.0", "指标计算（TSS/F1/AUC/混淆矩阵）"),
    ("matplotlib", "matplotlib", "3.4", "画训练曲线、混淆矩阵、ROC"),
    ("PIL", "Pillow", "8.0", "读取 600x600 PNG 磁图"),
]

# 可选依赖：没有也能跑，只是不能用某些辅助脚本
OPTIONAL = [
    ("tqdm", "tqdm", "4.0", "进度条（本项目未强制使用）"),
]

# 训练/测试必需的关键文件（与本脚本同级的相对路径）
REQUIRED_FILES = [
    ("训练集索引", "data/index/solar_flare_dataset_png_Lat60_Lon60_Th1000_train.csv"),
    ("验证集索引", "data/index/solar_flare_dataset_png_Lat60_Lon60_Th1000_val.csv"),
    ("测试集索引", "data/index/solar_flare_dataset_png_Lat60_Lon60_Th1000_test.csv"),
    ("PNG 磁图目录", "data/raw/png_600_Th1000"),
    ("Swin 预训练权重", "pretrained/swin_tiny_patch4_window7_224.pth"),
]


# ======================================================================
# 小工具
# ======================================================================

def _version_key(ver: str) -> tuple:
    """
    把 "2.14.1+cu132" 这类版本号转成可比较的元组 (2, 14, 1)。

    只取开头连续的"数字.数字.数字"，遇到非数字（如 +cu132、rc1）就停，
    这样预发布版/本地编译版也能比较。
    """
    parts: list[int] = []
    for chunk in str(ver).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _installed_version(import_name: str, pip_name: str) -> str | None:
    """
    取已安装版本号。优先用 importlib.metadata（不真正 import，速度快且不会触发
    CUDA 初始化），失败再退回 import 后读 __version__。
    """
    for name in (pip_name, import_name):
        try:
            return importlib_metadata.version(name)
        except Exception:
            continue
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "未知")
    except Exception:
        return None


def _is_installed(import_name: str) -> bool:
    """只用 find_spec 判断是否存在，不执行包内代码。"""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _print_header(title: str):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ======================================================================
# 各检查项
# ======================================================================

def check_python() -> bool:
    """打印 Python / conda 环境信息。"""
    _print_header("1. Python 与 conda 环境")

    print(f"  Python 版本    : {platform.python_version()}")
    print(f"  解释器路径     : {sys.executable}")
    print(f"  操作系统       : {platform.system()} {platform.release()} ({platform.machine()})")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_env:
        print(f"  conda 环境名   : {conda_env}")
        print(f"  conda 环境路径 : {conda_prefix}")
        if conda_env == "base":
            print("  [提示] 当前是 base 环境。如果项目专用环境是别的名字，")
            print("         请先 conda activate <环境名> 再重新运行本脚本。")

        # ---------- 关键检查：激活的环境 vs 实际执行的解释器 ----------
        # 这是最容易踩的坑：激活了 A 环境，但 PATH 里排在前面的 python 来自 B 环境。
        # 表现就是"我明明装了 torch，为什么 ModuleNotFoundError"。
        if conda_prefix:
            try:
                exe_ = Path(sys.executable).resolve()
                prefix_ = Path(conda_prefix).resolve()
                if prefix_ not in exe_.parents and exe_ != prefix_:
                    print()
                    print("  " + "!" * 70)
                    print("  [严重警告] 激活的环境与正在运行的 Python 不是同一个！")
                    print(f"             激活的环境 : {prefix_}")
                    print(f"             实际解释器 : {exe_}")
                    print("             这会导致\"明明装了包却 import 不到\"。")
                    print("             请确认 PATH 里 python 的优先级，或直接用绝对路径运行：")
                    print(f"             {prefix_ / 'python.exe'} scripts\\check_env.py")
                    print("  " + "!" * 70)
            except Exception:
                pass
    else:
        print("  conda 环境名   : (未检测到，可能没有激活 conda)")
        print("  [提示] 如果包检查出现大量缺失，很可能是没有激活正确的环境。")
        print("         先执行 conda activate <环境名>，再重新运行本脚本。")

    # Python 版本要求（项目用了 X | None 的类型注解，配合 future import 可向下兼容，
    # 但 3.9 以下不保证，故只做提醒）
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 9)
    if not ok:
        print(f"  [警告] Python {major}.{minor} 偏旧，建议 3.9 及以上（推荐 3.10+）。")
    else:
        print(f"  [OK] Python 版本满足要求（>= 3.9）")
    return ok


def check_packages() -> tuple[bool, list[str]]:
    """检查必需包，返回 (是否全齐, 缺失的 pip 包名列表)。"""
    _print_header("2. 依赖包检查")

    header = f"  {'状态':<6} {'包名':<16} {'已装版本':<18} {'最低要求':<10} 用途"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = True
    missing: list[str] = []
    too_old: list[tuple[str, str, str]] = []

    for import_name, pip_name, min_ver, purpose in REQUIRED:
        if not _is_installed(import_name):
            print(f"  {'[缺]':<6} {pip_name:<16} {'-':<18} {min_ver:<10} {purpose}")
            all_ok = False
            missing.append(pip_name)
            continue

        ver = _installed_version(import_name, pip_name) or "未知"
        need_ok = True
        if ver != "未知":
            need_ok = _version_key(ver) >= _version_key(min_ver)

        if need_ok:
            print(f"  {'[OK]':<6} {pip_name:<16} {ver:<18} {min_ver:<10} {purpose}")
        else:
            print(f"  {'[低]':<6} {pip_name:<16} {ver:<18} {min_ver:<10} {purpose}")
            all_ok = False
            too_old.append((pip_name, ver, min_ver))

    # ---------- 可选包 ----------
    print()
    print("  可选依赖（缺失不影响训练）：")
    for import_name, pip_name, min_ver, purpose in OPTIONAL:
        if _is_installed(import_name):
            ver = _installed_version(import_name, pip_name) or "未知"
            print(f"    [OK] {pip_name:<16} {ver:<18} {purpose}")
        else:
            print(f"    [--] {pip_name:<16} {'未安装':<18} {purpose}")

    # ---------- 安装建议 ----------
    if missing or too_old:
        print()
        print("  " + "-" * 74)
        print("  修复建议：")
        if missing:
            print(f"    缺少: {', '.join(missing)}")
            # torch / torchvision 要指定 CUDA 版本，单独给命令
            torch_like = [p for p in missing if p in ("torch", "torchvision")]
            others = [p for p in missing if p not in ("torch", "torchvision")]
            if torch_like:
                print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                print("      （cu121 适用于 Tesla P40 等 Pascal 卡；若本机是 RTX 30/40/50 系，")
                print("        可用更新的 cu124/cu126，见 https://pytorch.org/get-started/locally/）")
            if others:
                print(f"      pip install {' '.join(others)}")
        if too_old:
            for pip_name, ver, min_ver in too_old:
                print(f"    版本过低: {pip_name} 当前 {ver}，需要 >= {min_ver}")
                print(f"      pip install --upgrade {pip_name}")

    return all_ok, missing


def check_torch_cuda() -> bool:
    """
    检查 torch 的 CUDA 可用性，并针对不同架构给出训练配置建议。
    只有在 torch 装上之后才执行。
    """
    _print_header("3. GPU 与混合精度支持")

    if not _is_installed("torch"):
        print("  [跳过] torch 未安装，无法检查 GPU。")
        return False

    try:
        import torch  # noqa: PLC0415  延迟导入，避免包缺失时脚本崩溃
    except Exception as e:  # pragma: no cover
        print(f"  [失败] import torch 出错: {type(e).__name__}: {e}")
        return False

    print(f"  torch 版本     : {torch.__version__}")
    print(f"  编译时 CUDA    : {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA 可用      : {cuda_ok}")

    if not cuda_ok:
        print()
        print("  [警告] CUDA 不可用！可能原因：")
        print("         a) 没有 NVIDIA 显卡，或显卡驱动没装好")
        print("         b) 装的是 CPU 版 torch（版本号不带 +cuXXX）")
        print("         c) torch 的 CUDA 版本与驱动不兼容")
        print("         d) 【Pascal 卡（如 Tesla P40）常见】装的 torch 太新，")
        print("            CUDA 13 已不再支持 Pascal，需要用 cu121/cu124 的 torch。")
        return False

    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        major, minor = cap
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  显卡           : {name}")
        print(f"  计算能力       : {major}.{minor}")
        print(f"  显存           : {total_gb:.1f} GB")

        print()
        if major >= 8:
            print("  [建议] 该卡是 Ampere/Hopper/Blackwell 架构，支持 bf16：")
            print("         训练可加 --amp（bf16 混合精度，实测约 1.78 倍加速）")
        elif major >= 7:
            print("  [建议] 该卡是 Volta/Turing 架构：没有 bf16 硬件支持。")
            print("         训练请【不加】--amp，直接用 FP32。")
        else:
            print("  [建议] 该卡是 Pascal 架构（如 Tesla P40）：")
            print("         - 不支持 bf16")
            print("         - fp16 计算吞吐只有 FP32 的 1/64，比 FP32 还慢")
            print("         => 训练请【不要加】--amp，直接用 FP32。")
        return True
    except Exception as e:
        print(f"  [警告] 读取显卡信息失败: {type(e).__name__}: {e}")
        return False


def check_files() -> bool:
    """检查关键数据/权重文件是否存在（次要检查，不影响依赖结论）。"""
    _print_header("4. 关键文件（次要检查，换机器时数据可能还没拷过来）")

    all_ok = True
    for label, rel in REQUIRED_FILES:
        p = PROJECT_ROOT / rel
        if p.exists():
            if p.is_dir():
                n = sum(1 for _ in p.rglob("*.png"))
                print(f"  [OK] {label:<16} {rel}  ({n:,} 张 PNG)")
            else:
                size_mb = p.stat().st_size / 1024 ** 2
                print(f"  [OK] {label:<16} {rel}  ({size_mb:.1f} MB)")
        else:
            print(f"  [缺] {label:<16} {rel}")
            all_ok = False

    if not all_ok:
        print()
        print("  [提示] 文件缺失不影响\"依赖包是否齐全\"的结论。")
        print("         数据集和预训练权重需要单独拷贝，见 README 第五节。")
    return all_ok


# ======================================================================
# 主流程
# ======================================================================

def main() -> int:
    print("=" * 78)
    print("  环境自检 - 太阳耀斑预报项目（STCNet / JW-FD）")
    print("  只检测，不安装、不修改任何东西")
    print("=" * 78)

    py_ok = check_python()
    pkg_ok, missing = check_packages()
    cuda_ok = check_torch_cuda()
    files_ok = check_files()

    # ---------- 结论 ----------
    _print_header("结论")

    if pkg_ok:
        print("  依赖包：全部就绪 [OK]")
    else:
        print(f"  依赖包：有缺失 [缺] -> {', '.join(missing)}")

    if cuda_ok:
        print("  GPU   ：可用 [OK]")
    else:
        print("  GPU   ：不可用 [警告] -> 只能跑 CPU，训练会非常慢")

    print(f"  数据  ：{'齐全 [OK]' if files_ok else '有缺失（见第 4 节）'}")

    print()
    if pkg_ok and cuda_ok:
        print("  >>> 可以直接开始训练了：")
        print("        python main.py check        # 模型与数据自检")
        print("        python main.py check-data   # 数据集完整性校验")
        print("        python main.py train --smoke")
        print()
        print("  >>> 若确认无误，正式跑（Pascal 卡不要加 --amp）：")
        print("        python main.py train --tag try --epochs 3")
    elif pkg_ok and not cuda_ok:
        print("  >>> 依赖包齐了，但 GPU 不可用，请先解决 CUDA/驱动问题。")
        print("      参考 docs/notes/run_on_p40.md 第一节。")
    else:
        print("  >>> 请先按上面第 2 节的命令安装缺失的包，再重新运行本脚本。")

    print("=" * 78)

    # 退出码：0 = 依赖与 GPU 都 OK
    return 0 if (pkg_ok and cuda_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
