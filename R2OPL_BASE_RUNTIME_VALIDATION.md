# R²OPL-base 合并后正式入口验证（2026-09-20）

## 后续修复：Student 长序列诊断的显存峰值

服务器 Qwen3-4B → Qwen3-1.7B 在第二步 `compute_old_log_prob` 的 attention 中申请 294 MiB 失败；日志显示另一进程占用 87.34 GiB，但未提供足以确定该进程身份的记录。不能据此认定 attention、共享内存或模型权重本身是根因。

确认存在的遗漏是：R²OPL 的 Student top-16 诊断把 `prompt + response` 整段发给 vLLM 的 `prompt_logprobs`。训练端 fused logits 的分块不影响这个独立进程。vLLM V1 0.19.1、0.29.0 原实现一次对整个已调度 prompt 投影完整词表，并计算 log-softmax/top-k；正式配置允许一次 prefill 22528 个 token。

现在在已有 `monkey_patch_model` worker 初始化 RPC 中自动安装每块 128 token 的 prompt-score 实现，覆盖 Student 和 Teacher 的 V1 runner。每块完成后只保留紧凑的 CPU top-k 结果，并释放完整词表临时张量。保留 prefill 跨步缓存、完成边界、sampled-token rank、logits/logprobs 模式，以及 EOPD sampler hook；V2 保留其已有原生分块。该修复在 Linux 服务器同样生效，无需开启 WSL 开关，也无需改变正式配置的长度、batch、算法参数或 GPU 预算。

定向验证（当前验收环境为 `r2opl-cu12`）：

- vLLM 0.19.1 通过 14 项数值/生命周期测试，加上既有 R²OPL 回归共 29 passed；4 个不适用的 EOPD raw-logits 组合跳过。参考结果直接执行环境安装的上游方法。0.29.0 的先前对照同样通过，但后续只使用 0.19.1 验收。
- GPU 输出层对比使用 Qwen3-1.7B 的真实尺寸（hidden 2048、vocab 151936），top-16 token ID、概率及排名一致。此项为输出层定向测试，并非完整 1.7B 模型训练。

| prompt 计分 token 数 | 原路径峰值 allocated | 128-token 分块峰值 allocated |
| --- | --- | --- |
| 4096 | 6570 MiB | 820 MiB |
| 14336 | 21489 MiB | 900 MiB |

- 真实 Qwen3-0.6B vLLM 0.19.1 引擎连续处理两条 14336-token prompt-logprob 请求，使用正式 Student 相同的 `max_num_batched_tokens=22528`。两次均成功，观测到每次投影不超过 128 行，含模型和 3 GiB KV cache 的峰值 allocated 4648 MiB；启动和两次请求共 28.7 秒，第二次请求约 1.7 秒。该测试使用 eager 模式和单卡。
- `r2opl-cu12` 真实 `start_train.sh tests/configs/gsm8k/r2opl_base.yaml` 通过：两步各 8 条 rollout、评测 8 条、step 2 checkpoint，退出码 0，总耗时 184.3 秒。两步生成耗时 15.0 / 9.0 秒，actor 更新 6.27 / 5.97 秒，权重同步 2.42 / 2.53 秒；最低空闲显存 25221 MiB。测试使用已发布生产代码加本次补丁，未依赖其他算法未发布的修改。
- 上述测试均保留超过 3 GiB 的桌面显存；定向显存对比的最低观测空闲约 24.8 GiB。

验收依赖：Python 3.12.14、torch 2.10.0、vLLM 0.19.1、transformers 5.9.0、Ray 2.58.0、TransferQueue 0.1.7。完整训练结束清理时仍出现本地共享内存 `resource_tracker` 的 `/psm_*` 警告；训练产物核验和退出码均成功。本次不把清理警告当作服务器 OOM 的已证实根因。

后续所有实验和测试统一使用用户指定的 `r2opl-cu12`（vLLM 0.19.1），本地监控器会检查环境并在结果文件记录依赖版本。复现定向验证（项目根目录）：

```bash
conda activate r2opl-cu12
PYTHONPATH=.:verl python -m pytest tests/test_vllm_prompt_logprobs_chunking.py -q
PYTHONPATH=.:verl python tests/probe_r2opl_prompt_memory.py
PYTHONPATH=tests:.:verl python tests/probe_r2opl_long_prompt.py
python tests/run_gsm8k_formal.py r2opl_base
```

此修复消除了已复现的长序列完整词表临时显存峰值，但本地单卡无法证明服务器全部 8 卡进程的显存分配或长期 500-step 稳定性。原来的两步 1024-token 测试没有覆盖这条长序列内存风险，本节补充长序列验证范围。

## 首次合并修复的验证记录

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

当前重跑统一使用 `r2opl-cu12` 环境（首次记录使用的是 `r2opl`）。在项目根目录准备本地 `models/Qwen3-0.6B` 和 GSM8K question/answer JSON 数据后：

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
