# Table 4-1: ECE/AUC/LogLoss for 11 methods × 3 datasets

Reported: mean±std (ddof=1), N=3 seeds {1024, 2024, 3024}. ECE×100 for readability. M=100 bins.

**Note on statistical methods (platt/ir/hb)**: 这三种方法是数学上100% deterministic 算法（Platt=sklearn LogisticRegression lbfgs、IR=PAV 算法、HB=digitize+mean），跨 seed 输出完全唯一，因此 std=0 是算法本质，与 baiyimeng/UMC 原作一致。神经方法（umnn/neucalib/desc/sbcr/umc/umc_wor/uamcm/uamcm_wor）才报告真 3-seed 方差。

| Method | aliccp_ECE×100 | aliccp_AUC | aliccp_LogLoss | avazu_ECE×100 | avazu_AUC | avazu_LogLoss | criteo_ECE×100 | criteo_AUC | criteo_LogLoss |
|---|---|---|---|---|---|---|---|---|---|
| platt | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| ir | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| hb | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| umnn | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| neucalib | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| desc | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| sbcr | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| umc | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| umc_wor | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| uamcm | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| uamcm_wor | -- | -- | -- | -- | -- | -- | -- | -- | -- |

_Generated from experiments/runs/main (must pass sanity_check first)._
