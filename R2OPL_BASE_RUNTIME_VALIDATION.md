# R²OPL-base 合并后正式入口验证（2026-09-20）

## 正误分流与答案探针迁移纠正

后续逐条核验发现，下文早期两步运行只证明流程完成，不能证明算法正确：`explicit_step_prompt` 下训练判题被跳过，`rm_scores` 被填为 0，正确答案也进入错误 OPD 分支。早期“通过”的算法语义结论撤回；长序列分块的独立数值和显存验证仍有效。

本次以用户指定的 `OPD_Forge/algorithms/r2opl_base.py` 为算法依据（其内部实现名为 v2），连同 `utils/r2opl_v2.py` 和原答案探针构造代码迁移。目标公式保持不变：正确轨迹优势为 `difficulty * miu`，错误轨迹为 `difficulty * lambda * stop_gradient(log pi_T - log pi_S)`，其中 `difficulty = 1 - group_success`。全对组难度为 0，因此该组正确梯度为 0 是正常的；混合正误组必须产生正确分支梯度。

修复三个断点：依赖正误的训练强制判题；当前 `r2opl_base` 实际运行答案探针并携带 gold answer；控制器从 TransferQueue 显式读取探针结果，缺失时报错。探针沿用原始 token 前缀和最后句子边界，比较 `exp(tail_mean_logprob) - exp(head_mean_logprob) > 0.3`。根据用户确认，打分必须使用生成轨迹的 Student 当前权重；旧 OPD_Forge 辅助代码调用 Teacher 的部分已经改正，Teacher 仅提供 OPD 信号。通过的截断轨迹按 reward 1 计入 RL 分支和组难度；没有句子边界或未通过则保留 verifier 判定。新增 RPC/非有限值/上下文容量错误会终止并报告，避免静默失效。轨迹文件同时保存原判题分数及探针标记，可独立复核两者。

基础回归：移植原实现的目标函数、梯度、探针、agent loop、控制器测试，结合现有运行回归与判题、失败处理、因果位置及容量检查共 62 passed。使用 Student 自身探针后已重跑全部 62 项，agent loop 测试明确断言 Teacher 不得被用于截断探针。

真实 `start_train.sh` 验收均使用 `r2opl-cu12 / vLLM 0.19.1`：

| 场景 | 第 1 步 RL / OPD 轨迹 | 第 2 步 RL / OPD 轨迹 | 关键证据 | 总耗时 / 最低空闲显存 |
| --- | --- | --- | --- | --- |
| 标准 1024 长度、2 题 × 4 rollout | 8 / 0 | 5 / 3 | 第 2 步 reward=0.625，正确梯度 28.0186、错误梯度 0.0070003 | 196.9 秒 / 25395 MiB |
| 额外截断测试：生成上限 96 | 5 / 3 | 4 / 4 | 每步 8 条 Student 自探针；分别救援 5、4 条；正确梯度 39.6013 / 46.1192 | 190.4 秒 / 25145 MiB |

标准测试完成 2 步训练、更新权重、8 条评测和 checkpoint；生成结果逐条重判，与原 verifier 分数一致。第 1 步全对，难度为 0，因此正确梯度为 0 符合原算法。标准测试未触发截断，不能替代 Student 自探针验证。先前调用 Teacher 的截断测试记录仅作为排错历史，不计入最终验收。

最终 Student 自探针测试也完成全部两步、权重更新、8 条评测和 checkpoint。16 条训练轨迹的原 verifier 分数均为 0；Student 自探针分别把 5、4 条计入 RL，有效奖励均值为 0.625 / 0.5，错误分支梯度 0.0085663 / 0.0205884。逐条产物与控制器统计一致，两步混合正误组的正确梯度均非零。此结果来自真实 Student 引擎，未伪造探针分数。

梯度指标为整个分支完成 batch 累积、分布式同步后的全模型 L2 范数，已包含每轨迹 token 平均、全 batch 平均及 difficulty/miu/lambda 权重，记录于裁剪前；没有再按分支轨迹数单独平均。优化器使用两分支梯度的向量和。

额外截断复现：`python tests/run_gsm8k_formal.py r2opl_base --probe`，实际调用 `tests/configs/gsm8k/r2opl_base_probe.yaml`。Student 标准生成仍限制为 prompt 256 + response 768，本地 Student 引擎容量 1152 为额外自探针保留空间；服务器正式配置的 Student 引擎覆盖更长的评测上下文，已有训练自探针余量，无需改变启动命令。

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
