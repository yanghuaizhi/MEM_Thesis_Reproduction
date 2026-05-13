"""reproduction.analysis — 结果聚合与审计层。

子模块:
    sanity_check.py    M.0 内部质量门，不依赖论文数值（plan §A.6.1）
    diff_with_paper.py 四层验证 L1/L2/L3/L4 + 三态评估（plan §M.2-M.6）
    tables/            7 个 markdown 表生成器（plan §A.2）
    figures/           Ch3 图 3-1~3-4 + Ch4 图 4-1~4-2 生成器

工作流:
    1. 阶段 1-4 完成 → sanity_check 验证训练健康度
    2. 通过 sanity_check 后 → diff_with_paper 做四层论断对照
    3. tables + figures 生成论文回写素材
"""
