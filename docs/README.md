# docs/ — 复现项目文档（导师重点查阅）

本目录是给**论文导师审阅**的核心文档（plan §I.1）。导师不读代码，但会读这里的
方法学声明、结果摘要、差异审计、避坑清单。

## 目录索引

| 文档 | 用途 | 何时读 |
|------|------|--------|
| [01_paper_data_requirements.md](01_paper_data_requirements.md) | 论文每个数据点的来源映射（§A.2）| 复现前 |
| [02_methodology.md](02_methodology.md) | 11 方法的算法说明 + 引用 | 算法细节 |
| [03_setup_guide.md](03_setup_guide.md) | 环境 + 数据 + 容器配置 | 启动复现 |
| [04_experiment_protocol.md](04_experiment_protocol.md) | 8 阶段 SOP（plan §H）| 跑实验 |
| [05_results_summary.md](05_results_summary.md) | 主结果摘要（动态更新）| 看结论 |
| [06_paper_diff_audit.md](06_paper_diff_audit.md) | 与论文 v1.13 差异分析 | 决策 v1.14 修订 |
| [07_known_issues.md](07_known_issues.md) | 避坑清单 + 实际踩坑增量 | 排查问题 |
| [08_rtx5090_optimization.md](08_rtx5090_optimization.md) | Tier 1/2 硬件优化 | GPU 调优 |
| [09_local_ssh_workflow.md](09_local_ssh_workflow.md) | 本地-SSH 协作模式 | 远程操作 |
| [RUNBOOK.md](RUNBOOK.md) | 10 个故障应对剧本 | 出错时 |

## 推荐阅读顺序

**导师审阅路径**（不读代码）:
1. `../README.md` — 项目首屏 + 一键复现命令
2. `01_paper_data_requirements.md` — 论文数据需求
3. `04_experiment_protocol.md` — 实验执行流程
4. `05_results_summary.md` — 复现产物
5. `06_paper_diff_audit.md` — 差异分析 + v1.14 修订建议

**执行者复现路径**:
1. `03_setup_guide.md` → `04_experiment_protocol.md` → `RUNBOOK.md`
2. 跑完后 `05_results_summary.md` + `06_paper_diff_audit.md` 检查

## 与论文的对接

回写论文 v1.14 时，**所有数值改动**都源自 `06_paper_diff_audit.md` 的"v1.14 修订建议"
段落（不直接 `cp` 表格）。详见 plan §K.2。
