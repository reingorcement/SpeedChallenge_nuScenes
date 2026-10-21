# 历史归档

本分支保留整理前六个远程分支的完整目录快照，包含代码、配置、已入库权重和实验结果。每个 `snapshots/<名称>/` 的 tree 与对应原提交的根 tree 完全一致；不以“最新版本”替代其他快照。

| 快照 | 原方法/用途 |
|---|---|
| [main](snapshots/main) | 早期 CNN、原版 FlexiNet、初期单速度底盘融合 |
| [velocity-xy-fusion](snapshots/velocity-xy-fusion) | vx/vy 视觉与底盘融合、早期光流特征 |
| [dynkinvision-0703](snapshots/dynkinvision-0703) | KD/VKD，保留独有提交和独立旧版比较测试 |
| [dynkinvision-0718](snapshots/dynkinvision-0718) | V2 视觉几何及弱运动学修正 |
| [v2-seed42-recovery](snapshots/v2-seed42-recovery) | V2 续训修复、seed 42 结果及传感器故障实验 |
| [v3-visual-only](snapshots/v3-visual-only) | 历史 V3 整体工程，含继承的旧方法及实验记录 |

精确原分支名、提交号、tree、归档标签和文件数见 [SOURCE_MAP.json](SOURCE_MAP.json)。相同文件由 Git 对象复用，多个快照不重复上传相同权重。

## 恢复原版本

```bash
git fetch origin --tags
git switch -c restored-v2 archive/2026-10-21/feature/v2-seed42-recovery
```

也可以在本分支中进入单个快照原工程目录运行；旧路径、原环境依赖和已知限制均原样保留，归档不表示本次重新验证其精度或所有入口。

方法主线见 [main](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/main)。底盘辅助 V2 不归类成严格纯视觉。0703 独有内容没有混入 V3 覆盖同名文件，两个版本可分别追溯。

本分支的合并父提交保留 V3 和 0703 两条原历史；原 main 和其他分支均为其祖先。原 `v3-seed42-artifacts-20260729` 标签保持原指向。
