"""
----------------------------------------------------------------------
src/data/dataset.py
阶段二：PyTorch Dataset 与 DataLoader

本模块的作用：
    从 data/index/*.csv 读取样本，加载磁图 jpg + 10 个磁场参数，
    归一化后返回 ((image, params), label)，供 PyTorch 训练使用。

    返回值是「嵌套元组」((img, text), label)，这是为了兼容原代码
    models.py 里 Mynet.forward(inx) 的调用方式（inx[0] 是图，inx[1] 是参数）。

整体流程：
    1. SolarFlareDataset  ：PyTorch Dataset，负责单样本加载
    2. build_transform    ：构造训练/验证用的图像变换
    3. compute_param_range：在训练集上统计 10 个参数的 min/max
    4. build_weighted_sampler：构造加权采样器，平衡正负样本
    5. get_dataloaders    ：以上步骤的封装，返回三个 DataLoader

关键设计：
    - 单通道灰度图：HMI 磁图是 mode='L' 的灰度图，Swin 主干也按 in_chans=1 构建
    - 参数的 min/max 用训练集统计，验证/测试集复用（避免泄漏）
    - 数据增强只用于 train
    - 图片原始尺寸不一致（从 89×57 到 892×480 都有），统一 Resize 到 224×224

与论文的对应关系：
    论文 3.2 节："图像是单通道的灰度图像，图像在经过调整尺寸、随机水平翻转后的
    尺寸为 (224,224)，每块包含 4×4 个像素，每个像素有一个灰度值，
    因此展开后的维度为 4×4×1=16。"
    → 对应下面 IMG_SIZE=224 与 Swin 的 patch_size=4、in_chans=1
----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMG_SIZE,
    NUM_PARAMS,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
)


# ============================================================
# 1. Dataset
# ============================================================


class SolarFlareDataset(Dataset):
    """
    太阳耀斑二分类数据集（多模态：磁图 + 磁场参数）。

    每个样本：
        image  : (1, 224, 224) float32 tensor，单通道灰度图
        params : (1, 10)       float32 tensor，10 个归一化后的磁场参数
        label  : 0（48h 内无 M 级及以上耀斑）或 1（有）

    Parameters
    ----------
    csv_file : Path
        索引 CSV 路径（trainflare1.csv / validflare1.csv / testflare1.csv）。
    img_size : int
        输入图像边长，默认 224。
    param_min : np.ndarray | None
        10 个参数的最小值（用于 min-max 归一化）。None 表示不做归一化。
    param_max : np.ndarray | None
        10 个参数的最大值。
    transform : callable | None
        图像数据增强。只应该给 train 传，val/test 传 None。
    return_path : bool
        是否额外返回图片路径（调试和画错例时用）。

    说明
    ----
    CSV 列结构（无表头，13 列）：
        第 1 列    图片绝对路径
        第 2–11 列 10 个磁场特征参数
        第 12 列   标签
    由 src/data/preprocess.py 生成。
    """

    def __init__(
        self,
        csv_file: Path,
        img_size: int = IMG_SIZE,
        param_min: np.ndarray | None = None,
        param_max: np.ndarray | None = None,
        transform=None,
        return_path: bool = False,
    ):
        # header=None：CSV 没有表头
        self.df = pd.read_csv(csv_file, header=None)

        # 列数自检，防止读错文件（比如误读了别的 CSV）
        if self.df.shape[1] != NUM_PARAMS + 2:
            raise ValueError(
                f"{Path(csv_file).name} 列数应为 {NUM_PARAMS + 2}，"
                f"实际 {self.df.shape[1]}。请先运行 src/data/preprocess.py 生成索引。"
            )

        self.img_paths = self.df.iloc[:, 0].astype(str).tolist()
        # 第 2–11 列是参数（iloc 是 0-based，所以 1 : 11）
        self.params = self.df.iloc[:, 1 : 1 + NUM_PARAMS].to_numpy(dtype=np.float32)
        # 最后一列是标签
        self.labels = self.df.iloc[:, -1].to_numpy(dtype=np.int64)

        self.img_size = img_size
        self.transform = transform
        self.return_path = return_path

        # ---------- 参数归一化（min-max）----------
        # 为什么必须归一化：
        #   10 个参数的量纲差异极大，实测取值范围从 0 到 5.4e24
        #   （如 TOTPOT 量级 1e23，而 SHRGT45 只有 0~1）。
        #   不归一化的话，大数量级参数会主导 1D-CNN 的梯度。
        if param_min is not None and param_max is not None:
            span = np.where((param_max - param_min) > 0, param_max - param_min, 1.0)
            self.params = (self.params - param_min) / span
            # NaN 处理：实测原始数据第 8/9/10 列有缺失，归一化后仍是 NaN，
            # 这里填 0（= 归一化后的最小值），避免 loss 变 NaN。
            self.params = np.nan_to_num(self.params, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            self.params = np.nan_to_num(self.params, nan=0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ---------- 1. 加载磁图 ----------
        # HMI 磁图是 mode='L' 的单通道灰度图，尺寸不一致，交给 transform 统一
        img = Image.open(self.img_paths[idx])
        if img.mode != "L":
            # 双保险：万一遇到非灰度图，转成灰度，保证通道数与模型一致
            img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # 没有 transform 时用最基础的：Resize + ToTensor
            img = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )(img)

        # 确保是 (1, H, W)：ToTensor 对 mode='L' 会得到 (1, H, W)，这里再兜一次底
        if img.dim() == 2:
            img = img.unsqueeze(0)

        # ---------- 2. 磁场参数 ----------
        # 目标形状 (1, NUM_PARAMS)：第二维是"通道"，第三维是"长度"
        # 这样送进 nn.Conv1d(in_channels=1, ...) 才是合法的 3D 输入 [B, C, L]。
        # 注意：原代码在这里多套了一维变成 [N,1,10,1]（4 维），
        #       导致 Conv1d 直接报错，详见 docs/troubleshooting/bugs.md。
        params = torch.from_numpy(self.params[idx]).float().unsqueeze(0)  # (1, 10)

        # ---------- 3. 标签 ----------
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        # ---------- 4. 返回 ----------
        # 返回 ((img, params), label) 这种嵌套结构，
        # 是为了让 model((img, params)) 能直接解包，保持与原代码一致。
        if self.return_path:
            return (img, params), label, self.img_paths[idx]
        return (img, params), label


# ============================================================
# 2. 图像变换
# ============================================================


def build_transform(split: str, img_size: int = IMG_SIZE):
    """
    构造图像变换。

    Parameters
    ----------
    split : str
        'train' / 'val' / 'test'。
    img_size : int
        目标边长。

    Returns
    -------
    transform : torchvision.transforms.Compose

    设计说明
    --------
    训练集用 RandomResizedCrop + RandomHorizontalFlip（论文原文提到的增强）。
    验证/测试集只做确定性的 Resize + CenterCrop，保证可复现。

    Normalize([0.5], [0.5])：单通道写法。
    注意 list 长度必须等于通道数 1，写成 [0.5,0.5,0.5] 会报
    "output with shape [1,224,224] doesn't match the broadcast shape [3,224,224]"。
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # val / test：确定性变换
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


# ============================================================
# 3. 参数归一化统计量
# ============================================================


def compute_param_range(csv_file: Path = TRAIN_CSV) -> tuple[np.ndarray, np.ndarray]:
    """
    在训练集上统计 10 个磁场参数的 min / max。

    为什么必须用训练集统计
    ----------------------
    验证/测试集如果用自己的 min/max，相当于引入了测试集信息，
    属于数据泄漏，会让测试指标虚高。

    Parameters
    ----------
    csv_file : Path
        训练集 CSV。

    Returns
    -------
    (param_min, param_max) : tuple[np.ndarray, np.ndarray]
        长度均为 10 的数组。
    """
    df = pd.read_csv(csv_file, header=None)
    params = df.iloc[:, 1 : 1 + NUM_PARAMS].to_numpy(dtype=np.float32)

    # nanmin/nanmax：忽略 NaN（实测部分列有缺失）
    param_min = np.nanmin(params, axis=0)
    param_max = np.nanmax(params, axis=0)

    print(f"[STAT] 训练集参数取值范围（{len(df)} 条样本）:")
    for i, (lo, hi) in enumerate(zip(param_min, param_max)):
        print(f"         参数 {i + 1:2d}: [{lo:.4g}, {hi:.4g}]")

    return param_min, param_max


# ============================================================
# 4. WeightedRandomSampler
# ============================================================


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    根据标签分布构造加权采样器，让每个 batch 里正负样本接近 1:1。

    原理
    ----
    - 类越少，权重越大（1 / 类样本数）；
    - 每个样本的采样权重 = 它所属类的权重；
    - 采样时按权重随机抽取，少数类（耀斑）被抽中的概率更高。

    论文虽然对训练集做了负样本下采样，但下采样后仍是
    正 2988 : 负 5073（约 1:1.7），残余不平衡仍在，所以再加一层采样平衡。

    Parameters
    ----------
    labels : np.ndarray
        训练集标签数组。

    Returns
    -------
    sampler : WeightedRandomSampler
    """
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),   # 一个 epoch 采样多少次
        replacement=True,                   # 允许重复采样
    )

    print(f"[SAMPLER] 类别分布: 负={class_counts[0]}, 正={class_counts[1]}")
    print(f"[SAMPLER] 类别权重: 负={class_weights[0]:.6f}, 正={class_weights[1]:.6f}")
    return sampler


# ============================================================
# 5. DataLoader 工厂
# ============================================================


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    img_size: int = IMG_SIZE,
    use_augment: bool = True,
    use_weighted_sampler: bool = True,
    max_train_samples: int | None = None,
) -> dict:
    """
    构造 train / val / test 三个 DataLoader。

    Parameters
    ----------
    batch_size : int
    num_workers : int
        Windows 上建议 0。
    img_size : int
    use_augment : bool
        训练集是否用数据增强。做消融实验时可关掉。
    use_weighted_sampler : bool
        训练集是否用加权采样。
    max_train_samples : int | None
        只取训练集的前 N 条（冒烟测试用，正式训练传 None）。

    Returns
    -------
    loaders : dict
        {'train': DataLoader, 'val': DataLoader | None, 'test': DataLoader}
    """
    # ---------- 1. 统计参数的 min/max（只用训练集）----------
    param_min, param_max = compute_param_range(TRAIN_CSV)

    # ---------- 2. 构造三个 Dataset ----------
    train_ds = SolarFlareDataset(
        TRAIN_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
        transform=build_transform("train", img_size) if use_augment else build_transform("val", img_size),
    )

    # 冒烟测试：分层截断训练集（正负各取一半）
    #
    # 为什么不能直接 iloc[:N]：
    #   索引 CSV 里正样本排在前面（preprocess 是先写正样本再写负样本），
    #   直接取前 N 条会全是正样本，导致：
    #     1. WeightedRandomSampler 报"类别分布: 负=0, 正=200"，权重除零；
    #     2. 训练时只有一个类别，loss 与准确率都没有参考意义。
    #   所以这里按标签分层，各取 N/2 条。
    if max_train_samples is not None and max_train_samples < len(train_ds):
        n_each = max(1, max_train_samples // 2)
        idx_pos = np.where(train_ds.labels == 1)[0][:n_each]
        idx_neg = np.where(train_ds.labels == 0)[0][:n_each]
        keep_idx = np.concatenate([idx_pos, idx_neg])

        train_ds.img_paths = [train_ds.img_paths[i] for i in keep_idx]
        train_ds.params = train_ds.params[keep_idx]
        train_ds.labels = train_ds.labels[keep_idx]
        print(
            f"[SMOKE] 训练集已分层截断为 {len(train_ds)} 条 "
            f"(正 {len(idx_pos)} / 负 {len(idx_neg)})"
        )

    val_ds = None
    if Path(VAL_CSV).exists():
        val_ds = SolarFlareDataset(
            VAL_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
            transform=build_transform("val", img_size),
        )

    test_ds = SolarFlareDataset(
        TEST_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
        transform=build_transform("test", img_size),
    )

    # ---------- 3. 训练集 DataLoader ----------
    # persistent_workers：复用子进程，避免每个 epoch 重建（NUM_WORKERS>0 时才有意义）
    pw = PERSISTENT_WORKERS and num_workers > 0

    if use_weighted_sampler and len(train_ds) > 0:
        train_sampler = build_weighted_sampler(train_ds.labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,      # 用了 sampler 就不能再设 shuffle
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=pw,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=pw,
        )

    # ---------- 4. val/test：不 shuffle、不采样 ----------
    loaders = {
        "train": train_loader,
        "val": (
            DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=PIN_MEMORY,
                persistent_workers=pw,
            )
            if val_ds is not None
            else None
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=PIN_MEMORY,
            persistent_workers=pw,
        ),
    }

    print(
        f"[LOADER] train {len(train_ds)} 条"
        + (f", val {len(val_ds)} 条" if val_ds is not None else ", val 无")
        + f", test {len(test_ds)} 条 | num_workers={num_workers}"
    )
    return loaders


# ============================================================
# 6. 简单自测
# ============================================================

if __name__ == "__main__":
    loaders = get_dataloaders(batch_size=4)

    # 检查一个 batch 的形状
    (imgs, params), labels = next(iter(loaders["train"]))
    print()
    print("=" * 60)
    print("单 batch 形状检查")
    print("=" * 60)
    print(f"图像  : {tuple(imgs.shape)}  期望 (B, 1, 224, 224)")
    print(f"参数  : {tuple(params.shape)}  期望 (B, 1, 10)")
    print(f"标签  : {tuple(labels.shape)}")
    print(f"图像取值范围: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"参数取值范围: [{params.min():.3f}, {params.max():.3f}]")
    print(f"标签取值    : {labels.tolist()}")
