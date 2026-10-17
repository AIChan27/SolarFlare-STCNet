"""
----------------------------------------------------------------------
src/data/dataset.py
阶段二：PyTorch Dataset 与 DataLoader（JW-FD 版）

本模块的作用：
    从 data/index/*.csv 读取 JW-FD 的 66 列数据集表，加载 600×600 磁图 PNG
    + 29 个磁场特征，归一化后返回 ((image, params), label)，供 PyTorch 训练。

    返回值是「嵌套元组」((img, params), label)，这样 model((img, params))
    能直接解包，与 src/models/stcnet.py 的 Mynet.forward(inx) 接口一致。

与旧版（SOLAR-STORM2）的关键差异
--------------------------------
    1. CSV 有**表头**、共 66 列，不再是无表头的 13 列朴素格式：
           29 列特征 + 28 列二分类标签 + 7 列类别标签 + 2 列元数据
       所以这里改成**按列名**取数据（config.FEATURE_COLS / config.LABEL_COL），
       不再依赖硬编码的列号，避免列顺序变动导致静默错位。
    2. 特征数从 10 → 29，由 config.NUM_PARAMS 决定。
    3. 图片路径规则：CSV 的 image_path 形如 './3947/AR3947_20250103_202400.png'，
       需要拼到 config.JWFD_PNG_DIR 下（真实目录是 png_600_Th1000/3947/）。
       注意活动区子目录名是**裸数字**（3947），不是 'AR3947'。
    4. 三份划分（train/val/test）由官方 split_dataset.py 生成（AR 级 8:1:1，
       seed=62），**我们不再自己切分**，所以 preprocess.py 已作废。
    5. 图片是 600×600（不再是尺寸不一的 jpg），统一 Resize 到 224×224。

整体流程：
    1. SolarFlareDataset  ：PyTorch Dataset，负责单样本加载
    2. build_transform    ：构造训练/验证用的图像变换
    3. compute_param_range：在训练集上统计 29 个特征的 min/max
    4. build_weighted_sampler：构造加权采样器，平衡正负样本
    5. get_dataloaders    ：以上步骤的封装，返回三个 DataLoader

与论文的对应关系：
    论文 3.2 节："图像是单通道的灰度图像，图像在经过调整尺寸、随机水平翻转后的
    尺寸为 (224,224)，每块包含 4×4 个像素，每个像素有一个灰度值，
    因此展开后的维度为 4×4×1=16。"
    → 对应 IMG_SIZE=224 与 Swin 的 patch_size=4、in_chans=1
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
    FEATURE_COLS,
    LABEL_COL,
    COL_IMAGE_PATH,
    COL_IMAGE_FILENAME,
    JWFD_PNG_DIR,
    JWFD_DIR,
    JWFD_PNG_224_DIR,
    USE_PREPROCESSED_PNG,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
)


# ============================================================
# 0. 工具函数：把 CSV 里的相对路径解析成真实绝对路径
# ============================================================


def resolve_image_path(rel_path: str, filename: str | None = None) -> str:
    """
    把 CSV 里的 image_path 解析成真实图片的绝对路径。

    JW-FD 的 image_path 形如：
        './3947/AR3947_20250103_202400.png'
    真实文件位置：
        data/raw/png_600_Th1000/3947/AR3947_20250103_202400.png

    如果开启了 config.USE_PREPROCESSED_PNG（用 scripts/prepare_jwfd.py 预缩过），
    则映射到：
        data/processed/png_224/3947/AR3947_20250103_202400.jpg

    注意几个坑：
      - 前缀是 './'，不能用 lstrip('./')（lstrip 按字符集剥离，会误伤路径开头）
      - 活动区目录名是**裸数字** 3947，不是 'AR3947'
      - 万一路径里带反斜杠，需要统一成正斜杠

    Parameters
    ----------
    rel_path : str
        CSV 的 image_path 值。
    filename : str | None
        CSV 的 image_filename 值，作为兜底（当 image_path 解析失败时，
        从文件名里截出 AR 号再拼路径）。

    Returns
    -------
    path : str
        绝对路径字符串（即使文件不存在也返回，由调用方决定是否报错）。
    """
    raw = str(rel_path).replace("\\", "/").strip()

    # 从文件名或相对路径里截出 AR 号，供预缩目录使用
    fn = filename if filename else (Path(raw).name or None)
    ar_num = None
    if fn:
        head = str(fn).split("_")[0]            # 'AR3947'
        if head.upper().startswith("AR"):
            ar_num = head[2:]                   # '3947'
        else:
            ar_num = head

    # ---------- 优先用预缩后的 224×224 图片 ----------
    if USE_PREPROCESSED_PNG and ar_num and fn:
        stem = Path(str(fn)).stem
        dst = JWFD_PNG_224_DIR / ar_num / f"{stem}.jpg"
        return str(dst)

    # 绝对路径：直接用
    if raw.startswith("/") or (len(raw) > 2 and raw[1] == ":"):
        return str(Path(raw))

    # 去掉开头的 './'
    if raw.startswith("./"):
        raw = raw[2:]

    candidates = [
        JWFD_PNG_DIR / raw,        # 首选：png_600_Th1000 + 相对路径
        JWFD_DIR / raw,            # 次选：data/raw + 相对路径
    ]
    for c in candidates:
        if Path(c).exists():
            return str(c)

    # 兜底：从文件名（AR3947_20250103_202400.png）里截出 AR 号再拼
    if fn and ar_num:
        alt = JWFD_PNG_DIR / ar_num / str(fn)
        return str(alt)                          # 不存在也返回，便于报错时看清路径

    return str(JWFD_PNG_DIR / raw)


# ============================================================
# 1. Dataset
# ============================================================


class SolarFlareDataset(Dataset):
    """
    太阳耀斑二分类数据集（多模态：磁图 + 磁场特征）。

    每个样本：
        image  : (1, 224, 224)   float32 tensor，单通道灰度图
        params : (1, NUM_PARAMS) float32 tensor，归一化后的磁场特征
                 （JW-FD 时 NUM_PARAMS = 29）
        label  : 0（不满足标签条件）或 1（满足）

    Parameters
    ----------
    csv_file : Path
        JW-FD 的 66 列数据集 CSV（train/val/test 三份之一）。
    img_size : int
        输入图像边长，默认 224。
    param_min, param_max : np.ndarray | None
        29 个特征的 min/max（用于 min-max 归一化）。None 表示不做归一化。
    transform : callable | None
        图像数据增强。只应该给 train 传，val/test 传确定性变换。
    return_path : bool
        是否额外返回图片路径（调试和画错例时用）。
    label_col : str | None
        用哪一列作为标签。None 表示用 config.LABEL_COL。
        JW-FD 有 28 个二分类标签列，改这个参数即可切换任务。

    说明
    ----
    CSV 结构（有表头，66 列）：
        - 29 列特征：见 config.FEATURE_COLS
        - 28 列二分类标签：flare_label_{C1.0|M1.0|M5.0|X1.0}_{1|3|6|12|24|48|72}hr
        -  7 列类别标签：flare_class_{h}hr（GOES 等级字符串）
        -  2 列元数据：image_filename、image_path
    """

    def __init__(
        self,
        csv_file: Path,
        img_size: int = IMG_SIZE,
        param_min: np.ndarray | None = None,
        param_max: np.ndarray | None = None,
        transform=None,
        return_path: bool = False,
        label_col: str | None = None,
    ):
        self.csv_file = Path(csv_file)
        self.label_col = label_col or LABEL_COL

        # ---------- 只读需要的列，加快读取、降低内存 ----------
        need_cols = list(FEATURE_COLS) + [self.label_col]
        for c in (COL_IMAGE_PATH, COL_IMAGE_FILENAME):
            need_cols.append(c)

        df = pd.read_csv(self.csv_file, usecols=lambda c: c in set(need_cols))

        # ---------- 列完整性自检 ----------
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.csv_file.name} 缺少特征列: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}；"
                f"请检查 config.FEATURE_COLS 是否与 CSV 表头一致。"
            )
        if self.label_col not in df.columns:
            raise ValueError(
                f"{self.csv_file.name} 缺少标签列 {self.label_col!r}；"
                f"可用列形如 flare_label_M1.0_48hr，请检查 config.LABEL_COL。"
            )

        self.df = df
        self.n_rows = len(df)

        # ---------- 图片路径：预解析成绝对路径 ----------
        # 一次性算好（十几万条也就几十毫秒），避免每个 __getitem__ 都做字符串处理
        if COL_IMAGE_PATH in df.columns:
            rels = df[COL_IMAGE_PATH].astype(str).tolist()
            fns = (df[COL_IMAGE_FILENAME].astype(str).tolist()
                   if COL_IMAGE_FILENAME in df.columns else [None] * len(df))
            self.img_paths = [resolve_image_path(r, f) for r, f in zip(rels, fns)]
        elif COL_IMAGE_FILENAME in df.columns:
            fns = df[COL_IMAGE_FILENAME].astype(str).tolist()
            self.img_paths = [resolve_image_path("", f) for f in fns]
        else:
            raise ValueError(
                f"{self.csv_file.name} 既没有 {COL_IMAGE_PATH} 也没有 "
                f"{COL_IMAGE_FILENAME} 列，无法定位图片。"
            )

        # ---------- 特征与标签 ----------
        self.params = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
        labels = pd.to_numeric(df[self.label_col], errors="coerce").fillna(0)
        self.labels = labels.to_numpy(dtype=np.int64)

        self.img_size = img_size
        self.transform = transform
        self.return_path = return_path

        # ---------- 特征归一化（min-max）----------
        # 为什么必须归一化：
        #   29 个特征的量纲差异极大，实测例如
        #       Wavelet Energy L1 约 1e4~1e6，
        #       NL curvature 约 -1.6~4.7，
        #       Total unsigned flux 约 1e5~1e8。
        #   不归一化的话，大数量级特征会完全主导 1D-CNN 的梯度。
        #
        # FITS 分支的 12 个特征有 7.6% 缺失；PNG 分支完整，
        # 这里仍做 NaN 兜底（填 0，即归一化后的最小值）。
        if param_min is not None and param_max is not None:
            span = np.where((param_max - param_min) > 0, param_max - param_min, 1.0)
            self.params = (self.params - param_min) / span
            self.params = np.nan_to_num(self.params, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            self.params = np.nan_to_num(self.params, nan=0.0)

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        # ---------- 1. 加载磁图 ----------
        # JW-FD 的 PNG 是 600×600、mode='L' 的单通道灰度图
        img = Image.open(self.img_paths[idx])
        if img.mode != "L":
            # 双保险：万一遇到非灰度图，转成灰度，保证通道数与模型一致
            img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )(img)

        # 确保是 (1, H, W)：ToTensor 对 mode='L' 会得到 (1, H, W)，这里再兜一次底
        if img.dim() == 2:
            img = img.unsqueeze(0)

        # ---------- 2. 磁场特征 ----------
        # 目标形状 (1, NUM_PARAMS)：第二维是"通道"，第三维是"长度"
        # 这样送进 nn.Conv1d(in_channels=1, ...) 才是合法的 3D 输入 [B, C, L]。
        # 注意：原代码在这里多套了一维变成 [N,1,10,1]（4 维），
        #       导致 Conv1d 直接报错，详见 docs/troubleshooting/bugs.md。
        params = torch.from_numpy(self.params[idx]).float().unsqueeze(0)

        # ---------- 3. 标签 ----------
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        # ---------- 4. 返回 ----------
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
    验证/测试集只做确定性的 Resize，保证可复现。

    关于 Normalize([0.5], [0.5])
    ---------------------------
    这是单通道写法（list 长度必须等于通道数 1，写成 [0.5,0.5,0.5] 会报
    "output with shape [1,224,224] doesn't match the broadcast shape [3,224,224]"）。

    JW-FD 的 PNG 已把 0 G 定标到灰阶 127（即 127/255 ≈ 0.498），
    经过 Normalize(0.5, 0.5) 后 0 G 落在约 -0.004 ≈ 0，**正好居中**，
    所以这里无需修改，是自洽的。
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
# 3. 特征归一化统计量
# ============================================================


def compute_param_range(csv_file: Path = TRAIN_CSV) -> tuple[np.ndarray, np.ndarray]:
    """
    在训练集上统计 29 个磁场特征的 min / max。

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
        长度均为 NUM_PARAMS 的数组。
    """
    df = pd.read_csv(csv_file, usecols=lambda c: c in set(FEATURE_COLS))
    params = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)

    # nanmin/nanmax：忽略 NaN（FITS 分支有缺失，PNG 分支完整）
    param_min = np.nanmin(params, axis=0)
    param_max = np.nanmax(params, axis=0)

    print(f"[STAT] 训练集特征取值范围（{len(df)} 条样本，{len(FEATURE_COLS)} 个特征）:")
    for name, lo, hi in zip(FEATURE_COLS, param_min, param_max):
        print(f"         {name:<28s} [{lo:.4g}, {hi:.4g}]")

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

    JW-FD 的不平衡程度比 SOLAR-STORM2 更严重：
        主任务 flare_label_M1.0_48hr 训练集 正 10,834 / 负 148,101 ≈ 1:13.7
    所以加权采样器在这里更有必要。

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

    total = int(class_counts.sum())
    print(f"[SAMPLER] 类别分布: 负={class_counts[0]:,}, 正={class_counts[1]:,}"
          f"（正样本率 {class_counts[1] / max(total, 1):.2%}）")
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
    worker_init_fn=None,
    label_col: str | None = None,
) -> dict:
    """
    构造 train / val / test 三个 DataLoader。

    Parameters
    ----------
    batch_size : int
    num_workers : int
        本机实测 8 最优（16 逻辑核）。Windows 上必须配合 main 保护。
    img_size : int
    use_augment : bool
        训练集是否用数据增强。做消融实验时可关掉。
    use_weighted_sampler : bool
        训练集是否用加权采样。
    max_train_samples : int | None
        只取训练集的前 N 条（冒烟测试用，正式训练传 None）。
    worker_init_fn : callable | None
        传给 DataLoader 的子进程初始化函数，用于固定每个 worker 的随机种子。
    label_col : str | None
        用哪一列作为标签，默认 config.LABEL_COL。

    Returns
    -------
    loaders : dict
        {'train': DataLoader, 'val': DataLoader | None, 'test': DataLoader}
    """
    # ---------- 1. 统计特征的 min/max（只用训练集）----------
    param_min, param_max = compute_param_range(TRAIN_CSV)

    # ---------- 2. 构造三个 Dataset ----------
    train_ds = SolarFlareDataset(
        TRAIN_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
        transform=build_transform("train", img_size) if use_augment else build_transform("val", img_size),
        label_col=label_col,
    )

    # 冒烟测试：分层截断训练集（正负各取一半）
    #
    # 为什么不能直接 iloc[:N]：
    #   直接取前 N 条可能几乎全是负样本（正样本率仅 7.57%），
    #   导致 batch 内几乎没有正样本，loss 与指标没有参考意义。
    #   所以这里按标签分层，各取 N/2 条。
    if max_train_samples is not None and max_train_samples < len(train_ds):
        n_each = max(1, max_train_samples // 2)
        idx_pos = np.where(train_ds.labels == 1)[0][:n_each]
        idx_neg = np.where(train_ds.labels == 0)[0][:n_each]
        keep_idx = np.concatenate([idx_pos, idx_neg])

        train_ds.img_paths = [train_ds.img_paths[i] for i in keep_idx]
        train_ds.params = train_ds.params[keep_idx]
        train_ds.labels = train_ds.labels[keep_idx]
        train_ds.n_rows = len(keep_idx)
        print(
            f"[SMOKE] 训练集已分层截断为 {len(keep_idx)} 条 "
            f"(正 {len(idx_pos)} / 负 {len(idx_neg)})"
        )

    val_ds = None
    if Path(VAL_CSV).exists():
        val_ds = SolarFlareDataset(
            VAL_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
            transform=build_transform("val", img_size),
            label_col=label_col,
        )

    test_ds = SolarFlareDataset(
        TEST_CSV, img_size=img_size, param_min=param_min, param_max=param_max,
        transform=build_transform("test", img_size),
        label_col=label_col,
    )

    # ---------- 3. 训练集 DataLoader ----------
    # persistent_workers：复用子进程，避免每个 epoch 重建（num_workers>0 时才有意义）
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
            worker_init_fn=worker_init_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=pw,
            worker_init_fn=worker_init_fn,
        )

    # ---------- 4. val/test：不 shuffle、不采样 ----------
    loaders = {
        "train": train_loader,
        "val": (
            DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=PIN_MEMORY,
                persistent_workers=pw,
                worker_init_fn=worker_init_fn,
            )
            if val_ds is not None
            else None
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=PIN_MEMORY,
            persistent_workers=pw,
            worker_init_fn=worker_init_fn,
        ),
    }

    print(
        f"[LOADER] train {len(train_ds):,} 条"
        + (f", val {len(val_ds):,} 条" if val_ds is not None else ", val 无")
        + f", test {len(test_ds):,} 条 | num_workers={num_workers}"
    )
    return loaders


# ============================================================
# 6. 简单自测
# ============================================================

if __name__ == "__main__":
    loaders = get_dataloaders(batch_size=4, num_workers=0)

    (imgs, params), labels = next(iter(loaders["train"]))
    print()
    print("=" * 60)
    print("单 batch 形状检查")
    print("=" * 60)
    print(f"图像  : {tuple(imgs.shape)}  期望 (B, 1, 224, 224)")
    print(f"特征  : {tuple(params.shape)}  期望 (B, 1, {NUM_PARAMS})")
    print(f"标签  : {tuple(labels.shape)}")
    print(f"图像取值范围: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"特征取值范围: [{params.min():.3f}, {params.max():.3f}]")
    print(f"标签取值    : {labels.tolist()}")
