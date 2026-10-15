"""
----------------------------------------------------------------------
本包：src.data
数据 pipeline

阶段划分（见 docs/notes/dataset_build.md）：
    preprocess.py  阶段一：从 SOLAR-STORM2 原始数据构建训练/验证/测试索引 CSV
    dataset.py     阶段二：PyTorch Dataset 与 DataLoader

与参考项目 CME-Detection-Research 的对应关系：
    那个项目的数据 pipeline 拆成五阶段
    （download → preprocess → annotate → split → dataset），
    是因为它要自己下载 FITS、做校准/差分、再对照 CDAW 标注；
    本项目的数据集是官方已经整理好的（磁图 jpg + 参数表），
    所以只需要"构建索引"和"读数据"两步。
----------------------------------------------------------------------
"""
