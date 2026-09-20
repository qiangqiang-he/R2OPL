# PG-OPD / EOPD 本地正式入口验证

2026-09-20，WSL `r2opl-cu12`：Python 3.12.14、torch 2.10.0、vLLM 0.19.1、transformers 5.9.0、Ray 2.58.0、TransferQueue 0.1.7。

使用 `bash scripts/start_train.sh tests/configs/gsm8k/pg_opd.yaml` 和对应的 `eopd.yaml`，经过真实 Ray、vLLM、Teacher 打分、FSDP 更新、权重同步、评测及 checkpoint 保存。监控入口为 `python tests/run_gsm8k_formal.py pg_opd` / `eopd`。

单卡 Qwen3-0.6B，Teacher/Student 同模型；GSM8K 每批 2 题、每题 4 条 rollout、2 步；Student 总长度 1024，temperature/top_p 均为 1。监控在空闲显存低于 4GiB 或 180 秒无进展时停止。

| 算法 | 总耗时 | 两步生成耗时 | 最低空闲显存 | 结果 |
| --- | --- | --- | --- | --- |
| PG-OPD | 192.8 秒 | 17.0 / 8.0 秒 | 26235 MiB | 两步各 8 条，评测 8 条，保存成功 |
| EOPD | 192.3 秒 | 16.0 / 9.0 秒 | 25801 MiB | 两步各 8 条，评测 8 条，保存成功 |

评测输出逐条重新判题，与保存的分数一致。两者的训练目标不按验证奖励分流，显式步骤提示下训练奖励仍是可选诊断的占位值，不能把它当成训练准确率。

EOPD 修复：Teacher entropy hook 显式使用 V1 model runner；dense SDPA 的分块蒸馏对齐 Teacher 张量、entropy、标签和输出 padding。`tests/test_dense_chunked_distillation.py` 的 4 项测试对照完整投影的真实模型 loss 和梯度。

这些结果证明当前单卡、小模型、短序列执行链路；8 卡通信和正式大模型运行仍需服务器验证。R²OPL-base 的正误分流与答案探针修复独立验收，不能用此记录替代。
