# Table 5-5: u_mode 消融（PE / shuffled / logit）

u_mode=shuffled 时 u 与样本顺序完全解耦，理论上应消除 u 的方法学贡献。
AliCCP/Criteo 期望显著恶化（u 有效）；Avazu 期望不显著恶化（u 无独立贡献）。

| Dataset | PE ECE×100 | shuffled ECE×100 | logit ECE×100 | shuffled-PE 变化% |
|---|---|---|---|---|
| aliccp | -- | -- | -- | -- |
| avazu | -- | -- | -- | -- |
| criteo | -- | -- | -- | -- |
