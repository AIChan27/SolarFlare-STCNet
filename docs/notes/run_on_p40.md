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

在学校机器上**另建一个环境**，装 CUDA 12.x 的 torch（仍支持 sm_61）。例如：

```bash
# CUDA 12.1 版本，py3.10~3.12 都有
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

**期望看到**：`... True` / `Tesla P40 (6, 1)`

> ⚠️ 具体可用版本请以 `https://download.pytorch.org/whl/cu121` 上实际存在的为准；
> 关键判据是 **CUDA 版本 ≤ 12.x**（不要 13.x）。

---

## 二、🔴 第二条：**P40 上绝对不要加 `--amp`**

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

## 三、🟡 时间预估：先测一个 epoch 再定计划

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

## 四、🟡 数据搬过去：23 GB 还是 1.34 GB？

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

## 五、🟡 其他需要改的地方

### 5.1 `config.py` 里的 `NUM_WORKERS`

本地设的是 **8**（16 逻辑核）。学校机器的核数未知：

```bash
# Linux 下看核数
nproc
```

建议 `NUM_WORKERS = min(8, 核数 // 2)`。

### 5.2 路径

`config.py` 里所有路径都基于 `PROJECT_ROOT = Path(__file__).resolve().parent`，
**只要把整个项目目录拷过去，路径会自动正确**，不需要改任何路径。

但要确认这三个目录都在：

```
data/raw/png_600_Th1000/
data/index/solar_flare_dataset_png_Lat60_Lon60_Th1000_{train,val,test}.csv
pretrained/swin_tiny_patch4_window7_224.pth     ← 别忘这个，107 MB
```

### 5.3 先自检

```bash
python setup_project.py      # 目录 + 关键文件
python main.py check         # 环境 + 模型 + 数据
python main.py check-data    # 数据完整性 + 活动区零重叠
```

三个都过再开始训练。

### 5.4 Windows → Linux

- `main.py` 已有 `if __name__ == "__main__":` 保护，Linux 下 fork 启动子进程没问题
- 路径用 `pathlib.Path`，跨平台
- 前端脚本（`run_multiseed.py` 里的 `PYTHONUNBUFFERED`）在 Linux 下同样适用

---

## 六、P40 上的完整命令清单（**Windows 版**）

> 导师机器：**64 位 Windows 10 企业版 / Intel i7-10700 / Tesla P40 24G / 32G 内存**
> （来自论文 4.1 节实验环境）

### 6.0 与本地环境的对照

| 项 | 本地开发机 | 导师的 P40 机器 | 影响 |
|---|---|---|---|
| 系统 | Windows | **Windows 10 企业版** | ✅ 命令通用（都是 PowerShell） |
| CPU | 16 逻辑核 | **i7-10700 = 8 核 16 线程** | ✅ `NUM_WORKERS=8` 正好合适，**不用改** |
| 内存 | 32 GB | 32 GB | ✅ 够（本项目数据读取峰值约 3–5 GB） |
| GPU | RTX 5070 Laptop 12G (sm_120) | **Tesla P40 24G (sm_61)** | 🔴 架构差 6 代，见第一、二节 |
| CUDA | 13.2 | 需 **≤ 12.x** | 🔴 本地 torch 不能用 |

### 6.1 环境检查（PowerShell）

```powershell
# ① 看驱动与 CUDA 版本（这一步最关键）
nvidia-smi
#    关注右上角 "CUDA Version: xx.x" —— 这是驱动【最高支持】的 CUDA 版本
#    还要看 P40 当前是 TCC 还是 WDDM 模式（见 6.2）

# ② 看 Python 与 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
#    期望: (6, 1) 且 cuda_available=True
```

### 6.2 ⚠️ Windows 特有：P40 的 TCC / WDDM 模式

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

### 6.3 先自检

```powershell
python setup_project.py      # 目录 + 关键文件
python main.py check         # 环境 + 模型 + 数据
python main.py check-data    # 数据完整性 + 活动区零重叠
```

### 6.4 探针：先跑 1 个 epoch 量真实速度

```powershell
python main.py train --tag p40_probe --epochs 1
```

看日志末尾的 `xxxxs`，按下面的公式重算总时长：

```
总时长 ≈ 种子数 × (轮数 × 单轮秒数 + 300) / 3600   小时
```

### 6.5 正式 5 种子基准线（注意：**不要加 `--amp`**）

```powershell
python scripts\run_multiseed.py --tag jwfd --seeds 1 2 3 4 5 --epochs 3
```

让它在后台跑（新开一个 PowerShell 窗口执行即可，**不要关那个窗口**）。

### 6.6 看进度

```powershell
Get-Content "experiments\logs\multiseed_jwfd.log" -Tail 30 -Encoding UTF8
```

> 日志已加 UTF-8 BOM，`-Encoding UTF8` 或直接默认都能正常显示中文。

### 6.7 防止 Windows 睡眠中断训练

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

## 七、✅ 基准线配置已确定（2026-10-17 更新）

### 7.1 已完成的两个对照

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

### 7.2 另一个重要发现：**模型第 1 轮就到达峰值，之后全在过拟合**

不加权版本的三轮验证 AUC：**0.8966 → 0.8827 → 0.8782**（一路下降）
加权版本：0.8622（ep1）… 0.8712（ep8 最高）… 0.8578（ep13）

不管哪种采样器，**训练集准确率一路涨（0.934 → 0.942）而验证 AUC 不动甚至下降**。

**所以基准线不需要跑很多轮，3 轮足够**（早停会保存最优那轮）。

### 7.3 学校 P40 上的正式命令

```bash
# 前提：已按第一~五节准备好环境（CUDA 12.x 的 torch、数据、权重）

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

### 7.4 预计时间

| 平台 | 单 epoch | 5 种子 × 3 epoch |
|---|---|---|
| RTX 5070 Laptop（FP32） | 约 18 分钟 | 约 5 小时 |
| RTX 5070 Laptop（AMP bf16） | 约 11 分钟 | 约 3 小时 |
| **Tesla P40（FP32）** | **约 27–54 分钟（需实测）** | **约 7–14 小时** |

### 7.5 已知的局限（写论文时要如实说明）

1. **模型几乎立刻过拟合**：27.8M 参数 vs 158,935 训练样本，第 1 轮后泛化能力不再提升
2. **FAR 约 0.30**：每 10 次预警约有 3 次虚报，业务应用上偏高
3. **TSS 约 0.65**：比论文在 SOLAR-STORM2 上报告的 0.8251 低很多，
   但**两者不可比**（数据集、特征体系、划分方式、数据版本全不同；
   且本项目用的是更严格的**活动区级划分**，测试集活动区模型完全没见过）
4. **阈值必须靠验证集选**：模型输出概率对固定 0.5 阈值不友好（不加权时 TPR 仅 0.014）
