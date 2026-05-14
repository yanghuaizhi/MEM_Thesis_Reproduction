# 06: 与论文 v1.13 的差异审计（动态更新）

> 本文档跑完阶段 8 `bash scripts/generate_paper_artifacts.sh` 后自动汇总。
> 是 v1.14 论文修订的**唯一权威依据**（plan §K.2 关键纪律：禁止跳过本审计直接 cp 表格）。

## 0. 单位约定（CRITICAL，对比前必读）

**论文 v1.13 ECE 报告单位 = `×10³`**（ch4_method.md 4.6 节明确："ECE（Expected Calibration Error，**取 ×10³ 报告**），衡量预测概率与实际频率的平均偏差"）。

**复现 metrics.jsonl ECE 单位 = 原始小数**（UMC `utils/metric.py` 直接输出 0.0xxx 形式）。

**对比公式**：
```
论文报告 ECE × 10⁻³ ≈ 复现 metrics.jsonl ECE
```

具体例：
- 论文报告 "UAMCM AliCCP ECE 7.38±0.65" → 实际值 0.00738 ± 0.00065
- 复现 `experiments/runs/main/aliccp/uamcm/seed_1024/metrics.jsonl` `ece=0.007808` → ×10³ 后 = **7.81**
- 偏差 5.9%（在 1σ 内）→ supports

**`diff_with_paper.py` 必须做这个单位换算**，否则会误判 10x 偏差。

### 2026-05-14 16:00 AliCCP 三方对账示例（已验证）

| Method | 复现 ECE（小数）| 历史 v7_supp ECE（小数）| 论文 v1.13（×10³）| 偏差 |
|--------|---------------|----------------------|------------------|------|
| platt | 0.008119 | 0.008164 | 8.16 | -0.55% ✓ |
| ir | 0.007929 | 0.007960 | 7.96 | -0.39% ✓ |
| uamcm | 0.007808 | 0.007375 (3 seed mean) | 7.38±0.65 | +5.9% (1σ 内) ✓ |

LogLoss 偏差 < 0.1%、AUC 偏差 < 0.2%（这两个不需要单位换算，是原始数）。

## 1. 当前状态

复现未开始。本文档结构已就绪，待数据生成后由 `diff_with_paper.py` 自动填充。

## 2. 复现哲学（plan §A.4，必读）

**verify（独立验证），不是 reproduce-to-match（数值匹配）。**

- 论文论断 = **假设**，不是 ground truth
- 复现独立运行，跑完后客观对照（**支持 / 中立 / 反对**）
- 复现与论断矛盾 → 先怀疑复现严谨性（避坑清单 + 配置详表），再考虑修订论文

## 3. 评估框架

每个 P/S 论断的三态判定标准见 plan §A.4.1：
- supports: 满足主要支撑条件
- neutral: 主要条件部分满足
- opposes: 主要条件不满足
- no_data: 数据尚未生成

## 4. 总体结论（待数据）

```
Total verdicts: ?
  - supports: ?
  - neutral:  ?
  - opposes:  ?
  - no_data:  ?

Overall: ?
```

## 5. 各层详细对比

### L1 诊断（P1）— v9 数据
（自动填充，对应 `results/diff_audit/L1_diagnosis_verification.md`）

### L2 方法（P2/P3/P4）— main 数据
（自动填充，对应 `results/diff_audit/L2_method_verification.md`）

### L3 机制（P5）— v10 数据
（自动填充，对应 `results/diff_audit/L3_mechanism_verification.md`）

### L4 决策（S1/S2/S3）
（自动填充，对应 `results/diff_audit/L4_decision_verification.md`）

## 6. 论文 v1.14 修订建议（人工 review）

根据 §4-5 的自动判定，论文修订分三类:

### A. 轻微偏差（直接替换数值）
- 状态: supports 且复现数值与论文差 < ±0.5%
- 处理: `00_active/thesis/*.md` 中数字直接更新

### B. 中等偏差（文字微调）
- 状态: supports 但复现数值差 ±0.5% ~ ±2%
- 处理: 数字更新 + 章节末尾段落微调措辞（"约 14% 改善"）

### C. 显著差异（论断审视）
- 状态: opposes 或多个 neutral
- 处理: **暂停回写论文**，根因报告，与导师讨论是否修订论断

## 7. 数字溯源责任

每个写到论文 v1.14 中的数字必须能追溯到:
- `experiments/runs/.../metrics.jsonl` 行号
- 用何 `diff_with_paper` 函数计算
- 哪个 sanity_check 通过

不能溯源 = 不能写（plan §evidence-backed-analysis §数字溯源责任）。

## 8. 历史交叉验证

参考 `00_active/thesis/REVISION_LOG.md` §V1.1 — 论文当前 v1.13 数值已做过一次
交叉验证。本复现项目是**第二次独立验证**。

## 9. 与论文 commit 的对接

修订 v1.14 时，每个数字改动一个 commit：
```bash
git commit -m "ch4: update table 4-1 ECE means from reproduction (#ch4-tbl-4-1)"
```

详见 plan §K.2 Commit 规范。
