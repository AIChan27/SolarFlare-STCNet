"""
----------------------------------------------------------------------
src/train/trainer.py
训练循环与早停逻辑

训练流程：
    对每个 epoch：
        1. 训练一个 epoch（train_one_epoch）
        2. 在验证集上评估（可选）
        3. 记录 train_loss / train_acc / val_tss / ...
        4. 更新学习率（StepLR）
        5. 判断早停（基于监控指标，默认 TSS）
    训练结束：
        用 best_state 加载模型
        返回 history / best_state，测试集评估交给 src/train/evaluate.py

相比原代码 train.py 的改进：
    1. 分组学习率的三组参数过滤条件修正（原代码把 'bert' 写成遗留关键字，
       且第三组没排除 textcnn，导致部分参数被重复放进两个 param_group）
    2. 用 torch.save(model.state_dict()) 而不是 torch.save(model)
       （避免新版 PyTorch 的 weights_only 报错，也更省磁盘）
    3. 训练与评估解耦：本文件只负责训练，指标计算在 evaluate.py
    4. 保存最优权重而不是每个 epoch 都存一个 105 MB 的完整模型
----------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.supconloss import SupConLoss
from src.train.evaluate import evaluate


# ============================================================
# 随机种子
# ============================================================


def set_seed(seed: int):
    """
    设置全局随机种子，保证可复现。

    Parameters
    ----------
    seed : int
        任意整数。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 用确定性算法，代价是略慢
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_worker_seed(worker_id: int):
    """
    为 DataLoader 的每个子进程设置独立但可复现的随机种子。

    为什么需要这个函数
    ------------------
    num_workers > 0 时，数据加载（含 RandomResizedCrop、RandomHorizontalFlip
    等数据增强）发生在【子进程】里，而子进程不会继承主进程的随机状态。
    如果不处理，每个 worker 的随机序列由操作系统决定，导致：
        - 同一个 seed 跑两次，数据增强结果不同 → 结果不可复现
        - 多种子实验的"种子差异"里混进了"worker 随机性"，方差被高估

    做法：把主进程的 seed 和 worker_id 组合，保证
        同一次运行内各 worker 不同（避免增强撞车），
        不同运行间同种子可复现（满足实验要求）。

    Parameters
    ----------
    worker_id : int
        PyTorch 自动传入的 worker 编号（0, 1, 2, ...）。
    """
    import numpy as _np
    import random as _random
    import torch as _torch

    # 取主进程当前的种子（set_seed 已设过），与 worker_id 组合
    base_seed = _torch.initial_seed()
    worker_seed = (base_seed + worker_id) % (2 ** 32)

    _random.seed(worker_seed)
    _np.random.seed(worker_seed)
    _torch.manual_seed(worker_seed)

def get_device(prefer=None) -> torch.device:
    """
    选择计算设备。

    Parameters
    ----------
    prefer : str | None
        None 表示自动（有 CUDA 就用 CUDA）。

    Returns
    -------
    device : torch.device
    """
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 优化器与调度器
# ============================================================


def build_optimizer(model, config):
    """
    构造分组学习率的 AdamW 优化器。

    为什么要分组
    ------------
    Swin 主干用的是 ImageNet 预训练权重，只需要小幅微调（小学习率）；
    磁场参数分支的 1D-CNN 是随机初始化的，需要从头学（大学习率）。
    两者用同一个学习率会导致要么主干被破坏、要么 CNN 学得太慢。

    Parameters
    ----------
    model : Mynet
    config : Config

    Returns
    -------
    optimizer : torch.optim.AdamW

    修正说明
    --------
    原代码的分组过滤条件是：
        {'params': [... if 'textcnn_model' in n], 'lr': LR_CNN}
        {'params': [... if 'transformer' in n],  'lr': LR_SWIN}
        {'params': [... if 'transformer' not in n and 'bert' not in n]}   ← 问题在这
    第三个条件没有排除 textcnn，所以 CNN 分支的参数会同时出现在第 1 组和第 3 组。
    PyTorch 允许同一参数出现在多个 group 里，结果是学习率行为不确定。
    这里改成显式互斥的三个条件。
    """
    swin_params, cnn_params, other_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("transformer."):
            swin_params.append(p)
        elif name.startswith("text_cnn."):
            cnn_params.append(p)
        else:
            # 剩下的就是融合分类头 fc_1 等
            other_params.append(p)

    optimizer_grouped_parameters = [
        {"params": swin_params, "lr": config.lr_swin},
        {"params": cnn_params, "lr": config.lr_cnn},
        {"params": other_params, "lr": config.lr_other},
    ]

    print(
        f"[优化器] 分组学习率: Swin {len(swin_params)} 个参数组(lr={config.lr_swin}), "
        f"CNN {len(cnn_params)} 个(lr={config.lr_cnn}), "
        f"其他 {len(other_params)} 个(lr={config.lr_other})"
    )

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr_other)


def build_scheduler(optimizer, config):
    """
    构造学习率调度器：每 step_size 个 epoch 把学习率乘 gamma。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    config : Config

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.StepLR
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_lr_step, gamma=config.step_lr_gamma
    )


# ============================================================
# 单 epoch 训练
# ============================================================


def train_one_epoch(model, loader, optimizer, config, device, epoch: int):
    """
    训练一个 epoch。

    Parameters
    ----------
    model : Mynet
    loader : DataLoader
        训练集。每批返回 ((img, params), label)。
    optimizer : torch.optim.Optimizer
    config : Config
        用到 use_supcon / lambda_supcon / supoon_temperature。
    device : torch.device
    epoch : int
        当前轮数（仅用于打印）。

    Returns
    -------
    avg_loss : float
        平均损失（按样本数加权）。
    avg_acc : float
        训练集准确率。
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 对比损失模块（无参数，每批新建也可以，这里建一次复用）
    supcon = SupConLoss(temperature=config.supcon_temperature).to(device)

    # ---------- 混合精度（AMP）设置 ----------
    # 开关由 config.USE_AMP 控制，默认 False（保持与原实现完全一致的行为）。
    # 实测（本机 RTX 5070 Laptop，bs=32，含梯度裁剪）：
    #     FP32            : 1.00×（基准）
    #     AMP + bf16      : 约 1.5×
    #     AMP + fp16      : 约 1.5×
    # 之所以默认用 bf16 而不是 fp16：bf16 与 fp32 有相同的指数范围，
    # 不需要 GradScaler，也不容易出现 NaN（Swin 的注意力里有 exp）。
    use_amp = bool(getattr(config, "use_amp", False))
    amp_dtype_name = str(getattr(config, "amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    use_autocast = use_amp and device.type == "cuda"

    # ---------- GPU 能力检查（跨机器复现时很重要）----------
    # 不同架构对半精度的支持差异极大，换机器（比如拿到 Tesla P40 上跑）时必须判断：
    #
    #   compute capability    架构         bf16   fp16 计算吞吐
    #   ─────────────────────────────────────────────────────────
    #   6.x                   Pascal(P40)   ✗      1/64 of FP32（极慢！）
    #   7.0 / 7.5             Volta/Turing   ✗      有 Tensor Core，快
    #   8.x                    Ampere        ✓      快
    #   9.x / 12.x            Hopper/Blackwell ✓   快
    #
    # 结论：在 P40 上开 AMP 不但不会加速，fp16 反而比 FP32 慢；
    #      bf16 更是没有硬件支持（PyTorch 会退化成模拟，极慢或直接报错）。
    #      所以这里做一次能力检测，不支持就自动回退 FP32 并明确告知。
    if use_autocast:
        major, minor = torch.cuda.get_device_capability(device)
        dev_name = torch.cuda.get_device_name(device)

        if amp_dtype == torch.bfloat16 and major < 8:
            print(
                f"[AMP][警告] 当前 GPU 是 {dev_name}（compute capability {major}.{minor}），"
                f"不支持硬件 bf16（需要 Ampere/8.0 及以上）。\n"
                f"           已自动回退到 FP32 训练。若要半精度，请用 --amp-dtype fp16，"
                f"但在 Pascal（如 P40）上 fp16 吞吐只有 FP32 的 1/64，会更慢。"
            )
            use_autocast = False

        elif amp_dtype == torch.float16 and major < 7:
            print(
                f"[AMP][警告] 当前 GPU 是 {dev_name}（compute capability {major}.{minor}），"
                f"属 Pascal 架构，没有 Tensor Core，fp16 计算吞吐约为 FP32 的 1/64。\n"
                f"           开启 AMP 很可能【比 FP32 更慢】，建议去掉 --amp 直接用 FP32。"
            )

    # GradScaler 只在 fp16 下必要（bf16 指数范围与 fp32 相同，不需要缩放）
    use_scaler = use_autocast and amp_dtype == torch.float16
    #
    # 兼容性处理：torch.amp.GradScaler 这个"统一命名空间"版本是 PyTorch 2.3 才引入的。
    # 在 2.0~2.2（比如为了兼容 Tesla P40 而装的 cu121 版本）上只有 torch.cuda.amp.GradScaler。
    # 学校机器很可能装的是旧版 torch，所以这里做一次回退，避免 AttributeError。
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)          # PyTorch >= 2.3
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)             # PyTorch < 2.3

    for batch in loader:
        (imgs, params), labels = batch
        imgs = imgs.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 前向 + 分类损失：允许走半精度
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
            # 前向：返回 (图像模态特征, 分类概率)
            fea, probs = model((imgs, params))

            # ---------- 损失 ----------
            # 注意：模型最后一层已经是 softmax，输出的是概率。
            # 原代码写成 F.cross_entropy(outputs, labels)，
            # 而 cross_entropy 内部会自己做 log_softmax，
            # 相当于对已经 softmax 过的值再做一次 softmax（重复 softmax）。
            # 这里改用 F.nll_loss(log(probs))，数学上等价于对原始 logits 做交叉熵，
            # 只是数值上不如直接用 logits 稳，但对本任务够用且改动最小。
            loss_ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)

        # 对比损失强制走 fp32：它内部有 exp/log 和两两相似度，
        # 半精度下容易掉精度甚至触发 NaN（SupConLoss 里有 NaN 检查会直接抛错）。
        if config.use_supcon:
            with torch.autocast(device_type="cuda", enabled=False):
                loss_con = supcon(fea.float(), labels=labels)
            loss = loss_ce.to(loss_con.dtype) + config.lambda_supcon * loss_con
        else:
            loss = loss_ce

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)      # 裁剪前必须先 unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪，防止偶发的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---------- 统计 ----------
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


# ============================================================
# 完整训练流程
# ============================================================


def train(
    model,
    train_loader,
    val_loader,
    config,
    device,
    ckpt_path: Path | None = None,
    verbose: bool = True,
):
    """
    完整训练流程，含早停和最优权重保存。

    Parameters
    ----------
    model : Mynet
    train_loader : DataLoader
    val_loader : DataLoader | None
        None 表示不设验证集，此时不做早停，按训练 loss 保存最优。
    config : Config
    device : torch.device
    ckpt_path : Path | None
        最优权重保存路径（保存 state_dict）。
    verbose : bool

    Returns
    -------
    history : dict
        每轮的 train_loss / train_acc / val_* / lr。
    best_state : dict | None
        最优的模型 state_dict。
    """
    # ---------- 损失与优化器 ----------
    # 注意：模型内部已 softmax，所以这里不再传 nn.CrossEntropyLoss 给 forward 用，
    # 具体损失在 train_one_epoch 里算。
    if config.class_weights is not None:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_tss": [],
        "val_far": [],
        "val_auc": [],      # 阈值无关指标：评估"模型判别能力"，不受 0.5 阈值抖动影响
        "lr": [],
    }

    # 早停用的最优值：越大越好（loss 则越小越好）
    monitor = config.monitor_metric          # 'tss' / 'f1' / 'loss'
    best_score = -float("inf") if monitor != "loss" else float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # ---------- 1. 训练 ----------
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, config, device, epoch
        )

        # ---------- 2. 验证 ----------
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, threshold=config.eval_threshold)

        # ---------- 3. 记录 ----------
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["lr"].append(current_lr)

        if val_metrics is not None:
            history["val_loss"].append(None)          # 本实现不单独统计验证 loss
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_tss"].append(val_metrics["tss"])
            history["val_far"].append(val_metrics["far"])
            history["val_auc"].append(val_metrics["auc"])    # 阈值无关

        # ---------- 4. 学习率调度 ----------
        scheduler.step()

        # ---------- 5. 打印 ----------
        if verbose:
            msg = (
                f"Epoch {epoch:3d}/{config.num_epochs} | "
                f"loss={train_loss:.4f} acc={train_acc:.4f}"
            )
            if val_metrics is not None:
                msg += (
                    f" | val_AUC={val_metrics['auc']:.4f}"
                    f" val_ACC={val_metrics['accuracy']:.4f}"
                    f" val_F1={val_metrics['f1']:.4f}"
                    f" val_TSS={val_metrics['tss']:.4f}"
                    f" val_FAR={val_metrics['far']:.4f}"
                )
            msg += f" | lr={current_lr:.2e} | {time.time() - epoch_start:.0f}s"
            print(msg)

        # ---------- 6. 最优权重与早停 ----------
        if val_metrics is not None:
            score = val_metrics[monitor]
            improved = score > best_score if monitor != "loss" else score < best_score
        else:
            # 没有验证集时，用训练 loss 作为监控
            score = train_loss
            improved = score < best_score

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # 保存到磁盘（失败不影响训练继续）
            if ckpt_path is not None:
                try:
                    ckpt_path = Path(ckpt_path)
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, ckpt_path)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] 保存权重失败：{e}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(
                        f"\n早停：{monitor} 连续 {config.patience} 轮未提升，"
                        f"最佳 {monitor}={best_score:.4f}"
                    )
                break

    if verbose:
        print(f"\n训练结束，总耗时 {time.time() - start_time:.0f}s，"
              f"最佳 {monitor}={best_score:.4f}")

    return history, best_state
