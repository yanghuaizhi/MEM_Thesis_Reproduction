"""tests — reproduction 单元 + smoke 测试。

运行:
    pytest tests/                       # 全部
    pytest tests/test_utils.py -v       # 单文件
    pytest tests/ -k "not slow"         # 跳过 slow

测试覆盖（plan §M.1）:
    test_utils.py            seed / JsonlLogger / write_status
    test_configs.py          19 YAML 解析 + 一致性
    test_orchestrator.py     plan 生成 + dry-run + filter
    test_diff_with_paper.py  P1-P5 判定逻辑（mock data）
    test_metrics.py          ECE/AUC（依赖 UMC.utils.metric，本地可跑）
    test_path_param.py       smoke: UMC/_paths 解析（不依赖数据）
    test_tf32_drift.py       占位（需 GPU + 数据，远程跑）
"""
