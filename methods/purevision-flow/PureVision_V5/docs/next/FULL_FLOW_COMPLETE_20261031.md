# 全场景 CAM_FRONT 光流已完成

2026-10-31 13:01（Asia/Shanghai）实际核查：三个生成分片、train/dev打包和最终核验五个作业全部completed、exit_code=0，均只启动一次，无重启。新编码器训练尚未启动。

| 阶段 | 完成时间（北京时间） | 结果 |
|---|---|---|
| 全量生成 | 11:44:22 | 232239 / 232239 对，1000场景 |
| train/dev打包 | 12:41:22 | 原监督窗口所需158513对 |
| 最终核验 | 12:44:33 | 通过；验证范围见下文 |
| 完成报告发布 | 12:45:16 | GitHub远端提交已核实 |

范围为850个trainval场景和150个official test场景，仅CAM_FRONT；233239帧沿各自真实相机链构成232239相邻帧对。新计算222969对，严格核验并原字节复用原pilot的9270对。native+compact共 **1384739436318字节，约1.385TB十进制**；packed三个数组另有8765135232字节，约8.77GB。生成墙钟约12小时41分钟，合计约5.087对/s，主要受HDD存储限制。

RAFT前向输入1024×576，原生float32有符号像素位移，派生训练视图128×72、float16、向量缩放1/8。RAFT冻结，本阶段没有训练新的RAFT或测速模型，没有产生新的测速精度结果，也没有计算FB反向流。

## 数据位置与固定身份

缓存根：

```text
/mnt/sdb2/Workspace/zhangyu/dataset/nuscenes/derived/optical_flow/raft_large_C_T_SKHT_V2_1024x576_fwd_v1_20261030/
```

`v1.0-trainval/CAM_FRONT/`及`v1.0-test/CAM_FRONT/`分别保存原生与派生包；`packed_train_core_dev/`为后续训练可用的packed视图。它只包含原train_core/dev窗口，额外末尾边、两组holdout、15个未分配场景和official test不会进入监督训练或选模。标签及划分均未改变，PLANAR_PROXY和旧RGB权重的泛化限制仍然适用。

- 正式生成代码：`87d7468f911da5579bbbc39f79ec6bc85fa16dd6`，tag `v5-code-87d7468f911d`；固定worktree检查仍干净。
- run前缀：`20261030_full_nuscenes_front1024`。
- plan identity：`cbc303bbcd40f1189db741cab7cce6f3938d6a3be3e3326ce182555ce95cf620`。
- packed_index SHA256：`cce797035ce44b07e5f64ce0e560bfeb68a5f17bb1cb46ac8c16398a56782414`。
- 最终summary SHA256：`96946ac49805c1d4d57668bd0ff48db7cffa24371e2e277458dadcba8516b25c`。
- 完成结果提交：`8d581b75e908e0bde02d6c99853b97d9a4857a3a`，分支`results/purevision-v5-next`。

完整命令及每个作业状态在`/home/zhangyu/.local/share/speedest-v5-next/pipeline/20261030_full_nuscenes_front1024/`；日志在同ART根下`logs/20261030_full_nuscenes_front1024_*.log`；最终报告在`reports/full_flow_20261030_full_nuscenes_front1024/summary.json`及`.md`。

## 验证范围与限制

收尾程序独立重算三个分片的全部有序pair摘要，核验原元数据/group SHA、每个native/compact文件存在性及实际字节数；完整核对原train/dev pair集合、时间及所有packed数组SHA。

最终native/compact重载检查为每片首三对和末三对，共18对，检查字节SHA、来源、数值合同及精确派生值。**没有再次完整读取约1.385TB的全部native内容。** 新生成时沿用数值构造检查和native字节SHA绑定；复用/恢复逐对核验图像来源、合同及派生值。这里的“核验通过”指上述明确范围，不等于有真实光流标注的准确率评估。

本次前台核查再次确认packed索引SHA、已发布summary与服务器原JSON完全一致、五个作业终态与固定源码干净；没有无意义重跑全部生成或重复全量哈希。证据见 [终态独立检查](evidence/full_flow_terminal_inspection_20261031.json)。

## 巡视、发布与后续

正式半小时Astra共27次，全部成功；最近一次为12:30。所有任务完成后，服务器检查12:45自动停用巡视，完成发布器12:45核实远端提交后退出。13:01检查本用户crontab为空、tmux无残留会话、GPU无计算进程。原有GPU1非计算显存占用保留。

夜间GitHub曾出现SSH超时、部分小时快照延迟；未影响生成。最终完成报告已自动重试路径成功发布并核实，没有当前pending。辅助发布器v2修复记录保留在 [启动记录](FULL_FLOW_RUN_20261030.md) 和相邻证据中。

数据准备阶段已完成。下一阶段应在这份固定缓存及原train/dev协议上开展光流编码器训练与结构对照，继续遵守侧向验收和用户决策边界。不得重复启动已完成队列，也不能把本次缓存完成称为R02/Flex或后续融合训练完成。
