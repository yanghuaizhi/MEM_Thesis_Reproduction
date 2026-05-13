"""数据集预处理脚本（一次性工作）。

通用约定:
    输入:  <DATA_ROOT>/../raw/<dataset>/  原始数据
    输出:  <DATA_ROOT>/<dataset>/data.pkl  pandas DataFrame
           （columns: <稀疏特征...>, click；UMC get_data() 读取）

预处理逻辑参考 UMC/dataset/*_process.ipynb（保留作为权威 ground truth）。
本目录脚本是 ipynb 的脚本化版本，远程容器无 jupyter 时使用。

CLI:
    python -m reproduction.data.preprocess.aliccp
    python -m reproduction.data.preprocess.avazu
    python -m reproduction.data.preprocess.criteo
"""
