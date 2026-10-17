# 在 Tesla P40 上运行本项目的注意事项

> **背景**：本地开发机是 RTX 5070 Laptop（Blackwell, compute capability 12.0），
> 学校服务器是 Tesla P40（Pascal, compute capability **6.1**）。
> 两者架构相差 6 代，直接把本地环境搬过去会踩坑。
>
> 文档日期：2026-10-17

---

## 一、🔴 最严重的坑：**CUDA 13 不支持 Pascal，本地 torch 版本在 P40 上跑不起来**

### 问题

本地环境：

```
torch 2.14.1+cu132     ← CUDA 13.2
```

**CUDA 13.0 起已移除对 Maxwell / Pascal / Volta 的支持**（官方 release notes：
"CUDA 13.0 removes support for Maxwell, Pascal, and Volta architectures"）。

P40 是 **Pascal（sm_61）**，因此 `2.14.1+cu132` 这个 wheel **没有编译 sm_61 内核**，
在 P40 上会报类似：

```
NVIDIA GeForce ... is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_75 sm_80 sm_86 sm_90 ...
```

或者 `CUDA error: no kernel image is available for execution on the device`。

### 解决

在学校机器上**另建一个环境**，装 CUDA 12.x 的 torch（仍支持 sm_61）。

---

## 二、从零建一个干净的环境（★ 推荐做法）

> 决策：**只要磁盘够，就自己新建一个干净环境**，不去复用/改动别人已有的环境。
> 这样好处是：不污染导师机器上现有的环境、版本可控、出问题好回滚（直接删掉重建）。

### 2.1 先算磁盘预算

本项目所需包的实测体积（在本地 `cme` 环境上量的，Python 3.12 + Windows）：

| 项 | 体积 | 占比 |
|---|---|---|
| **torch**（含打包的 CUDA 库） | **2,880 MB** | **77.7%** |
| scipy | 109 MB | 2.9% |
| sympy（torch 依赖） | 66 MB | 1.8% |
| pandas | 60 MB | 1.6% |
| scikit-learn | 41 MB | 1.1% |
| numpy（含 numpy.libs） | 51 MB | 1.4% |
| matplotlib | 31 MB | 0.8% |
| torchvision | 18 MB | 0.5% |
| Pillow | 15 MB | 0.4% |
| 其余小依赖 | ~80 MB | 2.2% |
| Python 3.12 运行时 | 152 MB | 4.1% |
| **合计（新环境最终占用）** | **≈ 3.5 GB** | 100% |

**安装过程的磁盘峰值**：

| 阶段 | 占用 |
|---|---|
| torch wheel 下载 | 约 1.9 GB |
| 其余 wheel | 约 0.3 GB |
| 安装后环境本身 | 约 3.5 GB |
| **峰值（下载 + 最终）** | **≈ 5.5–6 GB** |

> **结论：按 6 GB 预留磁盘**（若用 `--no-cache-dir`，装完会回落到约 3.5 GB）。
> 另外 conda 自己会额外写一份 `pkgs` 缓存（约 0.2–0.5 GB），可用 `conda clean -a` 清掉。

### 2.2 建环境（完整命令）

```powershell
# 1) 建一个干净的 Python 3.12 环境（只装 Python 本身，约 150 MB）
conda create -n jwfd python=3.12 -y
conda activate jwfd

# 2) 用 pip 装包，并且不缓存（省约 2 GB 磁盘）
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir numpy pandas scikit-learn matplotlib pillow

# 3) 验证
python scripts\check_env.py
```

**第 3 步的期望输出**（关键三行）：

```
  依赖包：全部就绪 [OK]
  GPU   ：可用 [OK]           ← 显卡应显示 Tesla P40，计算能力 6.1
  数据  ：齐全 [OK]
```

退出码 `0` = 全部就绪。如果 GPU 那行显示"不可用"，见下面 2.4 的排查。

### 2.3 三个"为什么"

**① 为什么用 pip 装 torch，而不是 `conda install pytorch`？**

| 安装方式 | 环境体积 | 原因 |
|---|---|---|
| **pip**（推荐） | **约 3.5 GB** | CUDA 库打包在 torch wheel 内部，**只有一份** |
| `conda install pytorch` | **5–8 GB** | conda 会额外把 `cudatoolkit`、`cudnn`、`mkl` 作为**独立包**再装一份，与 torch 内部那套重复 |

**② 为什么加 `--no-cache-dir`？**

pip 默认会把下载过的 wheel 存到 `%LOCALAPPDATA%\pip\Cache`，torch 那个 wheel 有 **1.9 GB**。
装完不删就是白占 2 GB。用 `--no-cache-dir` 直接不缓存。
（如果以后还要重装、想复用，可以不加，但记得事后 `pip cache purge`。）

**③ 为什么 pin `python=3.12`？**

本项目在 3.12 上验证通过；3.13/3.14 上部分包的预编译 wheel 可能还没跟上，
容易出现"源码编译失败"或性能问题。

### 2.4 常见问题排查

| 现象 | 原因与处理 |
|---|---|
| `check_env.py` 显示 `[严重警告] 激活的环境与正在运行的 Python 不是同一个` | PATH 里 python 优先级不对。用绝对路径跑：`& "$env:CONDA_PREFIX\python.exe" scripts\check_env.py` |
| `依赖包：全部就绪` 但 `GPU：不可用` | 看 `nvidia-smi` 右上角 `CUDA Version`。若 < 12.0，说明驱动太老，需要升级驱动（见第七节） |
| 报 `no kernel image is available` | 装成了 cu13x 的 torch。卸载重装 cu121：`pip uninstall -y torch torchvision` 再重装 |
| `pip install` 卡住或超时 | 换国内镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`（但 **torch 的 cu121 必须走 `download.pytorch.org`**，镜像上没有） |
| 建完发现装错了，想重来 | `conda env remove -n jwfd -y`，然后从 2.2 重来（干净、无残留） |

### 2.5 如果磁盘实在紧张（备选方案）

| 方案 | 最终占用 | 代价 |
|---|---|---|
| 正常装（推荐） | 3.5 GB | — |
| 装 **CPU 版** torch（`--index-url .../whl/cpu`） | **约 0.9 GB** | ❌ 不能用 GPU 训练，只能跑通代码逻辑 |
| 复用导师机器上已有的 torch 环境 | 0 | 需要先 `conda env list` 看有没有，且**不能乱动别人的环境** |

> 只有在"磁盘 < 1 GB"或者"只是想先把代码跑通"的情况下才考虑 CPU 版。

---

## 三、🔴 P40 上绝对不要加 `--amp`

### 硬件事实

| 项 | Tesla P40 | RTX 5070 Laptop |
|---|---|---|
| 架构 | Pascal (sm_61) | Blackwell (sm_120) |
| FP32 算力 | **12 TFLOPS** | 显著更高 |
| FP16 算力 | **0.183 TFLOPS（= FP32 的 1/64）** | 有 Tensor Core，远高于 FP32 |
| Tensor Core | ❌ 没有 | ✅ 有 |
| bf16 硬件支持 | ❌ 没有 | ✅ 有 |
| 显存 | 24 GB HBM2 | 12 GB GDDR7 |

### 结论

| 配置 | P40 上的效果 |
|---|---|
| FP32（默认） | ✅ **唯一正确选择** |
| `--amp`（bf16） | ❌ 没有硬件支持，PyTorch 退化成模拟 → 极慢或报错 |
| `--amp --amp-dtype fp16` | ❌ 没有 Tensor Core，fp16 吞吐是 FP32 的 **1/64** → **比 FP32 慢得多** |

**所以 P40 上的命令是：去掉 `--amp`。**

代码里已加自动保护（`src/train/trainer.py`）：检测到 compute capability < 8.0
且要求 bf16 时，会**打印警告并自动回退 FP32**，不会静默变慢：

```
[AMP][警告] 当前 GPU 是 Tesla P40（compute capability 6.1），不支持硬件 bf16
           （需要 Ampere/8.0 及以上）。已自动回退到 FP32 训练。
```

---

## 四、🟡 时间预估：先测一个 epoch 再定计划

P40 的 FP32 算力只有 12 TFLOPS，实测很可能比 5070 Laptop 慢 **1.5–3 倍**。

| 平台 | 单 epoch（bs=32, FP32） | 5 种子 × 6 epoch |
|---|---|---|
| RTX 5070 Laptop | 约 18 分钟 | 约 10 小时 |
| **Tesla P40（估计）** | **约 27–54 分钟** | **约 14–28 小时** |

### 建议做法

**先只跑 1 个 epoch，量出真实速度，再决定种数和轮数：**

```bash
python main.py train --tag p40_probe --epochs 1
```

跑完看日志里的 `xxxxs`，然后按下面的公式重算：

```
总时长 ≈ 种子数 × (轮数 × 单轮秒数 + 300 秒测试) / 3600   小时
```

> 别凭本地经验估算 —— P40 的实测数据和笔记本可能差很多。

---

## 五、🟡 数据搬过去：23 GB 还是 1.34 GB？

数据集有两种形态：

| 形态 | 体积 | 训练时会不会额外缩放 |
|---|---|---|
| `data/raw/png_600_Th1000/` 原始 PNG | **23 GB** | 会（每张做 600→224） |
| `data/processed/png_224/` 预缩 JPEG | **1.34 GB** | 不会（已经是 224） |

### 方案 A：搬原始 23 GB（**推荐用于正式实验**）

- 优点：与本地实验**完全一致**，数据增强（`RandomResizedCrop` 作用在 600×600 上）不变
- 缺点：拷贝慢

### 方案 B：只搬预缩的 1.34 GB（**适合先跑通流程**）

```bash
# 本地先跑（约 26 分钟 / 4 进程）
python scripts/prepare_jwfd.py
# 然后把 data/processed/png_224/ 拷到学校机器
# 并改 config.py:  USE_PREPROCESSED_PNG = True
```

- 优点：体积小 17 倍，拷贝很快
- ⚠️ **缺点**：`RandomResizedCrop(224)` 会作用在**已经缩到 224 的图**上，
  而不是原始 600×600。增强效果与本地基线**不完全一致**，严格来说不能直接和本地结果混着比。

> **建议**：正式定基准线用方案 A（23 GB）；只是想先验证环境跑通，用方案 B。

---

## 六、🟡 其他需要改的地方

### 6.1 `config.py` 里的 `NUM_WORKERS`

本地设的是 **8**（16 逻辑核）。学校机器的核数未知：

```bash
# Linux 下看核数
nproc
```

建议 `NUM_WORKERS = min(8, 核数 // 2)`。

### 6.2 路径

`config.py` 里所有路径都基于 `PROJECT_ROOT = Path(__file__).resolve().parent`，
**只要把整个项目目录拷过去，路径会自动正确**，不需要改任何路径。

但要确认这三个目录都在：

```
data/raw/png_600_Th1000/
data/index/solar_flare_dataset_png_Lat60_Lon60_Th1000_{train,val,test}.csv
pretrained/swin_tiny_patch4_window7_224.pth     ← 别忘这个，107 MB
```

### 6.3 先自检

```bash
python setup_project.py      # 目录 + 关键文件
python main.py check         # 环境 + 模型 + 数据
python main.py check-data    # 数据完整性 + 活动区零重叠
```

三个都过再开始训练。

### 6.4 Windows → Linux

- `main.py` 已有 `if __name__ == "__main__":` 保护，Linux 下 fork 启动子进程没问题
- 路径用 `pathlib.Path`，跨平台
- 前端脚本（`run_multiseed.py` 里的 `PYTHONUNBUFFERED`）在 Linux 下同样适用

---

## 七、🟡 驱动要不要更新？

### 7.1 先看一行输出，就知道有没有必要动

```powershell
nvidia-smi
```

右上角会显示 **`CUDA Version: xx.x`**。注意：这**不是**你装的 CUDA 工具包版本，
而是**当前驱动最高支持的 CUDA 运行时版本**。

| nvidia-smi 显示的 CUDA Version | 结论 |
|---|---|
| **12.x**（如 12.2 / 12.4 / 12.6） | ✅ **完全不用动驱动**，直接装 cu121/cu124 的 torch |
| **11.x** | ⚠️ 想用 CUDA 12 的 torch 就**必须先升级驱动** |
| **13.x** | ⚠️ 驱动很新。注意新版驱动可能已不再支持 Pascal，反而要**降级** |
| 报错 / 无输出 | 🔴 驱动没装或显卡未被识别，先解决驱动问题 |

### 7.2 驱动版本与 CUDA 的最低对应关系

torch 的 cu12x wheel 只要求**驱动不低于某个下限**，不需要恰好匹配：

| torch 的 CUDA 版本 | Windows 驱动最低要求（约） |
|---|---|
| cu121 | **≥ 531** |
| cu124 | **≥ 551** |

所以只要 `nvidia-smi` 显示的 CUDA Version ≥ 12.1，基本都能直接装 cu121 而不动驱动。

### 7.3 为什么**不推荐**主动更新驱动

1. **P40 是 Pascal 架构，已进入停止支持周期**
   NVIDIA 已公布 Maxwell / Pascal / Volta 的支持终止计划
   （[GeForce 支持计划公告](https://nvidia.custhelp.com/app/answers/detail/a_id/5676)，
   [中文报道](https://m.expreview.com/100975.html)）。
   **装太新的驱动可能反而让 P40 无法使用。**
2. **研究服务器原则是"能跑就别动"** —— 驱动更新要重启，可能黑屏、
   可能导致 CUDA 不匹配、可能影响机器上别人正在跑的任务。
3. **驱动只要能满足 CUDA 运行时下限就够了**，装更新不会让训练变快。

### 7.4 如果确实需要更新（比如现在是 CUDA 11.x）

| 要点 | 说明 |
|---|---|
| 装**数据中心 / Tesla 驱动分支** | 不要装 GeForce Game Ready 驱动（那是给消费级显卡的） |
| **不要跳到最新大版本** | 去 NVIDIA 驱动下载页选 `Tesla P40`，看它列出哪些兼容版本 |
| **先记录当前版本** | 把 `nvidia-smi` 输出截图/存档，便于回滚 |
| **挑没人在用的时候** | 更新需要重启电脑 |
| 更新后立刻验证 | 见 7.5 |

### 7.5 更新后的验证

```powershell
nvidia-smi                                                          # 确认能识别 Tesla P40，CUDA Version >= 12.1
python -c "import torch; print(torch.cuda.get_device_capability(0))"  # 应输出 (6, 1)
python scripts\check_env.py                                         # 应显示 GPU：可用 [OK]
```

> **一句话建议**：先跑 `nvidia-smi`。**只要 CUDA Version ≥ 12.1，就完全不要碰驱动。**

---

## 八、P40 上的完整命令清单（**Windows 版**）

> 导师机器：**64 位 Windows 10 企业版 / Intel i7-10700 / Tesla P40 24G / 32G 内存**
> （来自论文 4.1 节实验环境）

### 8.0 与本地环境的对照

| 项 | 本地开发机 | 导师的 P40 机器 | 影响 |
|---|---|---|---|
| 系统 | Windows | **Windows 10 企业版** | ✅ 命令通用（都是 PowerShell） |
| CPU | 16 逻辑核 | **i7-10700 = 8 核 16 线程** | ✅ `NUM_WORKERS=8` 正好合适，**不用改** |
| 内存 | 32 GB | 32 GB | ✅ 够（本项目数据读取峰值约 3–5 GB） |
| GPU | RTX 5070 Laptop 12G (sm_120) | **Tesla P40 24G (sm_61)** | 🔴 架构差 6 代，见第一、三节 |
| CUDA | 13.2 | 需 **≤ 12.x** | 🔴 本地 torch 不能用 |

### 8.1 环境检查（PowerShell）

```powershell
# ① 看驱动与 CUDA 版本（这一步最关键）
nvidia-smi
#    关注右上角 "CUDA Version: xx.x" —— 这是驱动【最高支持】的 CUDA 版本
#    还要看 P40 当前是 TCC 还是 WDDM 模式（见 8.2）

# ② 看 Python 与 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
#    期望: (6, 1) 且 cuda_available=True
```

### 8.2 ⚠️ Windows 特有：P40 的 TCC / WDDM 模式

Tesla P40 **没有显示输出接口**，必须靠主板/核显输出画面（i7-10700 自带 UHD 630，正好可以）。

| 模式 | 说明 | 对本项目的影响 |
|---|---|---|
| **WDDM**（默认，Windows 显示驱动模型） | 和桌面系统共享显存管理，有额外的驱动开销 | 训练能跑，但**性能略差**，且大显存占用时可能被系统回收 |
| **TCC**（Tesla 计算模式） | 纯计算模式，无显示开销 | **推荐**，性能更好、更稳定 |

查看当前模式：

```powershell
nvidia-smi --query-gpu=name,driver_model.current --format=csv
```

若要切到 TCC（**需要显示器接在核显上、且要重启**）：

```powershell
# 需要在管理员 PowerShell 中执行
nvidia-smi -i 0 -dm 1      # 1 = TCC, 0 = WDDM
# 然后重启电脑
```

> ⚠️ 切换前先确认系统能从核显正常输出画面，否则切完可能黑屏。
> **如果现在能正常训练，不建议折腾这个** —— 先跑通再说。

### 8.3 先自检

```powershell
python setup_project.py      # 目录 + 关键文件
python main.py check         # 环境 + 模型 + 数据
python main.py check-data    # 数据完整性 + 活动区零重叠
```

### 8.4 探针：先跑 1 个 epoch 量真实速度

```powershell
python main.py train --tag p40_probe --epochs 1
```

看日志末尾的 `xxxxs`，按下面的公式重算总时长：

```
总时长 ≈ 种子数 × (轮数 × 单轮秒数 + 300) / 3600   小时
```

### 8.5 正式 5 种子基准线（注意：**不要加 `--amp`**）

```powershell
python scripts\run_multiseed.py --tag jwfd --seeds 1 2 3 4 5 --epochs 3
```

让它在后台跑（新开一个 PowerShell 窗口执行即可，**不要关那个窗口**）。

### 8.6 看进度

```powershell
Get-Content "experiments\logs\multiseed_jwfd.log" -Tail 30 -Encoding UTF8
```

> 日志已加 UTF-8 BOM，`-Encoding UTF8` 或直接默认都能正常显示中文。

### 8.7 防止 Windows 睡眠中断训练

```powershell
# 查看当前设置
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE

# 插电时永不睡眠、电池时永不睡眠
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change monitor-timeout-ac 0
```

**必须确保机器不会睡眠**，否则训练直接中断（本项目不支持中途 resume 训练，只能重跑该种子）。

---

## 九、✅ 基准线配置已确定（2026-10-17 更新）

### 9.1 已完成的两个对照

| 项目 | 加权采样（旧默认） | **不加权（新默认）** |
|---|---|---|
| 实验 tag | `jwfd_seed1`（seed 1） | `jwfd_nows`（seed 42） |
| epoch | 13（早停） | 3 |
| 验证集 AUC | 0.8712 | **0.8966** |
| 测试集 AUC | 0.8567 | **0.8803** |
| 测试集 TSS | 0.5893 | **0.6545** |
| 测试集 FAR | 0.3278 | 0.3010 |
| 测试集 F1 | 0.3449 | 0.3763 |
| val 最优阈值 | 0.10 | **0.010** |

**结论：关掉加权采样器（`USE_WEIGHTED_SAMPLER = False`）明显更好**，已设为新默认。

> ⚠️ 注意这里两次实验的随机种子不同（1 vs 42），严格来说不是完全同条件对照。
> 但 AUC 提升 0.024 远大于历史观测的种子间标准差（约 0.003），可信度较高。

### 9.2 另一个重要发现：**模型第 1 轮就到达峰值，之后全在过拟合**

不加权版本的三轮验证 AUC：**0.8966 → 0.8827 → 0.8782**（一路下降）
加权版本：0.8622（ep1）… 0.8712（ep8 最高）… 0.8578（ep13）

不管哪种采样器，**训练集准确率一路涨（0.934 → 0.942）而验证 AUC 不动甚至下降**。

**所以基准线不需要跑很多轮，3 轮足够**（早停会保存最优那轮）。

### 9.3 学校 P40 上的正式命令

```bash
# 前提：已按第一~六节准备好环境（CUDA 12.x 的 torch、数据、权重）

# 探针：先跑 1 个 epoch 量速度
python main.py train --tag p40_probe --epochs 1

# 正式 5 种子基准线（注意：不要加 --amp！）
nohup python scripts/run_multiseed.py \
      --tag jwfd --seeds 1 2 3 4 5 --epochs 3 \
      > multiseed_run.log 2>&1 &

# 看进度
tail -f experiments/logs/multiseed_jwfd.log
```

**注意**：`run_multiseed.py` 内部调用 `main.py test` 时已强制带 `--val-threshold`，
所以阈值协议（验证集选、测试集报告）会被自动遵守，不会泄漏。

### 9.4 预计时间

| 平台 | 单 epoch | 5 种子 × 3 epoch |
|---|---|---|
| RTX 5070 Laptop（FP32） | 约 18 分钟 | 约 5 小时 |
| RTX 5070 Laptop（AMP bf16） | 约 11 分钟 | 约 3 小时 |
| **Tesla P40（FP32）** | **约 27–54 分钟（需实测）** | **约 7–14 小时** |

### 9.5 已知的局限（写论文时要如实说明）

1. **模型几乎立刻过拟合**：27.8M 参数 vs 158,935 训练样本，第 1 轮后泛化能力不再提升
2. **FAR 约 0.30**：每 10 次预警约有 3 次虚报，业务应用上偏高
3. **TSS 约 0.65**：比论文在 SOLAR-STORM2 上报告的 0.8251 低很多，
   但**两者不可比**（数据集、特征体系、划分方式、数据版本全不同；
   且本项目用的是更严格的**活动区级划分**，测试集活动区模型完全没见过）
4. **阈值必须靠验证集选**：模型输出概率对固定 0.5 阈值不友好（不加权时 TPR 仅 0.014）
