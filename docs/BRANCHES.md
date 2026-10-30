# 分支与恢复

| 长期分支 | 职责 |
|---|---|
| main | 方法总览和三个方法目录的整合集合 |
| baseline/cnn-image-only | 普通 CNN 图像基线 |
| baseline/flexinet-image-only | 原版 FlexiNet 和明确标记的图像回归派生版 |
| method/purevision-flow | V3、补偿版、V4 独立实现 |
| method/purevision-v5 | V5 S2 六组特征融合实验；独立开发 worktree，训练从已推送 commit 的 detached worktree 启动 |
| archive/legacy | 原六分支完整快照及来源记录 |

三个方法分支和 main 使用一致的 `methods/<路线>/` 路径，未来仅提交所属目录的修改，再通过普通合并进入 main。不能用某方法分支的目录整体覆盖 main。

原分支全部以 `archive/2026-10-21/<原分支名>` 标签保留（包括 feature/ 前缀）。既有 `v3-seed42-artifacts-20260729` 标签不改动。

```bash
git fetch origin --tags
git switch -c restore-v2 archive/2026-10-21/feature/v2-seed42-recovery
```

上述恢复只创建本地分支。`archive/legacy` 的 `snapshots/` 下每个快照与原提交根 tree 完全一致；各快照保留自己的相对目录关系。原预测 CSV、图表和权重即使不出现在方法目录，也仍在原标签及对应快照中。

原 main 是整合提交的祖先，更新采用正常前进；不改写历史。0703 独有提交由归档分支和标签共同保留。本轮 V5 保留全部既有分支和用户未提交工作，不删除旧分支、不改写历史、不强制推送。

V5 的固定训练 commit 为 `3979268699ce6db46d536cf22dc16f02b969a465`，标签 `v5-s2-ready-v0.3`；完整结果在 `v5-s2-results-v0.1`。最新 [GitHub 审核导航](../methods/purevision-flow/PureVision_V5/AI_REVIEW.md) 指向交接、原计划和小型工件凭证。文档分支前进不改变已完成实验的 detached 工作目录。
