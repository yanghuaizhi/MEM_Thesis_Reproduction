"""30_reproduction 复现工作层。

子包说明:
    configs/     YAML 配置中心（数据集/方法/实验/硬件/路径）
    data/        数据下载 + 预处理脚本
    analysis/    结果聚合（tables/figures）+ sanity_check + diff_with_paper
    utils/       seed / logging / gpu / status 工具

注意:
    本包**不依赖 UMC/**。UMC/ 是算法层（被调方），可以 import 本包；
    本包通过 subprocess 或 sys.path 注入调用 UMC/{pretrain,train_*}.py。
"""
