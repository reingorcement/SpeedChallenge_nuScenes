# S23几何训练约束 A/B

状态：partial；固定提交 `c09c602f1296994336c8d079435f835a7ccaa462`。



本轮比较原速度监督与加入离线姿态、冻结道路语义及几何先验的训练约束；部署输入和S23结构保持一致。

固定 λ=0.445245457，只由pilot train初始梯度标定一次。



| scope/组 | best轮 | best dev | last dev | best train | dev−train |

|---|---:|---:|---:|---:|---:|

| PILOT_A | 12 | 0.393976 | 0.418796 | 0.18418606899496023 | 0.20978968382371846 |

| PILOT_B | 11 | 0.392940 | 0.430024 | 0.2084383582172161 | 0.18450120113293964 |



pilot B−A：ΔEPE=-0.001036 m/s（-0.263%）；改善/退化场景=3/5。

10–15m/s：{'count': 306, 'metrics': {'epe': {'delta': -0.12354403179441303, 'relative_percent': -24.243647536351627}, 'mae_vx': {'delta': -0.12235277150970653, 'relative_percent': -24.168126069773894}, 'mae_vy': {'delta': -0.01499564965176426, 'relative_percent': -38.336830983412575}, 'rmse_vx': {'delta': -0.12439217745613351, 'relative_percent': -21.52968003007635}, 'rmse_vy': {'delta': -0.015818945821964137, 'relative_percent': -34.12096151942297}, 'bias_vx': {'delta': 0.1634868640525668, 'relative_percent': -33.201633180547134}, 'bias_vy': {'delta': 0.04361175921033411, 'relative_percent': -122.15199480278703}}}。

前三正贡献场景：[('scene-0402', 0.01770889129439034), ('scene-0048', 0.0016543915333642797), ('scene-0584', 0.0015025790881561046)]；占净收益比例=20.13703285126271（其他场景退化抵消时可超过100%）。



pilot自动扩大判断：passed；{'geometric_preflight': True, 'same_initialization': True, 'finite_training': True, 'actual_geometric_coverage': True, 'no_invalid_projected_depth': True, 'within_tolerance': True}。1%是筛选容忍度。



完整两轴、速度段、全部场景及非零/正负/近零侧向结果见CSV；侧向充分性沿用原门槛。

历史原S23 seed42 best19 dev≈0.178472，TAIL_FT≈0.178169，仅作协议不同的历史参照。



- Single seed42; no cross-seed stability claim.

- Offline orientations, frozen pretrained semantics and geometric priors are extra uncertain training supervision.

- PLANAR_PROXY is ground-projected rear-axle yaw-plane velocity, not CG or certified true sideslip.

- All dev windows are the primary metric; only training has a geometric cache. Dev geometry-valid-region attribution is unavailable.

- Above15m/s dev count remains11; repeated predictions are not independent labels.
