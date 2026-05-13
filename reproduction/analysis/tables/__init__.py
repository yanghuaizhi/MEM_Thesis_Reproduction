"""tables 子包 — 各章节表格生成器。

每个 *.py 都可独立 CLI 运行，输出 results/tables/<name>.{md,csv}。

文件 → 论文表的映射（plan §A.2）:
    table_4_1.py     表 4-1 ECE/AUC/LogLoss 全部方法（11 方法 × 3 数据集）
    table_4_2.py     表 4-2 主结果改善百分比 + seed 一致性
    table_4_3.py     表 4-3 统计方法 vs 神经方法对比
    table_4_4.py     表 4-4 UMC 系列消融
    tables_5_3_5_6.py  Ch5 三重门槛 + shuffled-u 消融
"""
