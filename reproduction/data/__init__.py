"""reproduction.data — 数据下载与预处理子包。

子模块:
    download   下载工具 + md5 校验 + retry（aria2c 包装）
    preprocess 三个数据集的预处理脚本

数据流向:
    raw/ → preprocess → processed/<dataset>/data.pkl
        UMC/train_neu_*.py 内 get_data() 读 data.pkl 做 60/20/20 split

详见 30_reproduction/data/README.md
"""
