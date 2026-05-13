# Layer L3 (P5 + 派生预判) verification

summary: supports=0, neutral=0, opposes=0, no_data=6

## N/A P5 [aliccp] shuffled-u: **no_data**
  - detail: v10 missing (pe=False, shuffled=False)

## N/A P5 [criteo] shuffled-u: **no_data**
  - detail: v10 missing (pe=False, shuffled=False)

## N/A P5 [avazu] shuffled-u: **no_data**
  - detail: v10 missing (pe=False, shuffled=False)

## N/A P5 派生预判 vs 实验 [aliccp]: **no_data**
  - detail: v9 samples NPZ 缺失或 u 字段未派生

## N/A P5 派生预判 vs 实验 [avazu]: **no_data**
  - detail: v9 samples NPZ 缺失或 u 字段未派生

## N/A P5 派生预判 vs 实验 [criteo]: **no_data**
  - detail: v9 samples NPZ 缺失或 u 字段未派生


---

## 独立性标注（plan §A.4 复现哲学）

v9 派生预判 + v10 实验验证均基于**同一 model**（UAMCM × PackedDeepFM × seed=1024），因此本节命中率验证的是"复现内部 v9/v10 一致性"，**不是用不同 model 的真独立验证**。如需真独立，应用 model A 派生预判 + model B 验证（超出当前 reproduction 范围）。
