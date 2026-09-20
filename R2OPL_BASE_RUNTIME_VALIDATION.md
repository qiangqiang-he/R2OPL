# R²OPL-base 合并后正式入口验证（2026-09-20）

已在 WSL Conda `r2opl` 完成真实 `start_train.sh` 在线训练：2 个 step，每步 2 题 × 4 条 rollout，最后评测 8 条并保存 step 2 checkpoint。进程退出码 0，总耗时 210.6 秒（含启动），最低空闲显存 21516 MiB。第二步生成耗时约 10 秒。此记录不表示其他算法或服务器 8 卡已完成实测。

## 本次修复

- 修正 R²OPL 分支反向传播的合并缩进错误，以及各分支 microbatch 的梯度同步。
- 移除诊断请求误传到 vLLM RPC 的 `routing_key`。
- 修正 TransferQueue 0.1.7 空元数据构造、复用 rollout logprob 时丢失 batch 返回值。
- rollout 异常保留原始错误并传回训练端，等待同题其他任务结束后报告失败，避免空 batch 掩盖根因。
- 兼容 vLLM 0.29 的 tied embedding/lm_head 分桶加载验证；接收端权重加载失败会回传错误，避免发送端一直等确认。
- 关闭 vLLM sleep 时不再调用 wake_up。
- Gemma 31B Teacher 正式配置改为独立 4 卡 TP=4、1 个副本；Student 仍使用另外 4 卡。

本地测试显式启用 Teacher/Student 同卡及 CPU 共享内存传权重，两个开关默认均为 false，正式配置仍走独立 Teacher 池和 CUDA IPC。服务器显存充足也不能消除 API 不兼容；目前未证实服务器 `/dev/shm` 不足。

## 复现

在 `r2opl` 环境、项目根目录，准备本地 `models/Qwen3-0.6B` 和 GSM8K question/answer JSON 数据后：

```bash
python tests/run_gsm8k_formal.py r2opl_base
```

监控器实际调用 `bash scripts/start_train.sh tests/configs/gsm8k/r2opl_base.yaml`。总长度 1024（prompt 256 + response 768），温度/top-p 均为 1。每 2 秒检查显存，剩余低于 4GiB 时停止，为桌面保留至少 3GiB 的余量；180 秒无进展则停止排查。完成后核对两步日志、每步 8 条输出、8 条评测及 checkpoint。

CPU 回归：`tests/test_r2opl_merge_runtime.py`，15 passed。覆盖实际安装的 vLLM loader、共享内存传输成功/失败、Qwen/Gemma 小模型前后向、R²OPL 分支梯度、RPC、ReplayBuffer 和正式配置。分布式通信在 CPU 测试中模拟；实际 8 卡通信和大模型显存峰值仍需服务器验证。

验证环境：Python 3.12.13、torch 2.13.0+cu130、vLLM 0.29.0、transformers 5.10.4、Ray 2.58.0、TransferQueue 0.1.7。

服务器更新本仓库后，继续使用原启动命令：

```bash
bash scripts/start_train.sh configs/r2opl_base/r2opl_base_gemma4_31b_it_to_e2b_it_len12k_lambda0p05_miu16_500steps.yaml
```

如服务器依赖版本与上述环境不同，应先核对版本；本地两步测试用于检查执行链路，不用于判断模型训练效果。
