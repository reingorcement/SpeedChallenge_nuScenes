# S23几何训练约束 A/B

状态：partial；固定提交 `c09c602f1296994336c8d079435f835a7ccaa462`。



本轮比较原速度监督与加入离线姿态、冻结道路语义及几何先验的训练约束；部署输入和S23结构保持一致。

固定 λ=0.445245457，只由pilot train初始梯度标定一次。



| scope/组 | best轮 | best dev | last dev | best train | dev−train |

|---|---:|---:|---:|---:|---:|



完整两轴、速度段、全部场景及非零/正负/近零侧向结果见CSV；侧向充分性沿用原门槛。

历史原S23 seed42 best19 dev≈0.178472，TAIL_FT≈0.178169，仅作协议不同的历史参照。



- Single seed42; no cross-seed stability claim.

- Offline orientations, frozen pretrained semantics and geometric priors are extra uncertain training supervision.

- PLANAR_PROXY is ground-projected rear-axle yaw-plane velocity, not CG or certified true sideslip.

- All dev windows are the primary metric; only training has a geometric cache. Dev geometry-valid-region attribution is unavailable.

- Above15m/s dev count remains11; repeated predictions are not independent labels.
