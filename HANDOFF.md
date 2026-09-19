# R²OPL 开发交接

更新时间：2026-09-19 22:11（Asia/Shanghai）。本文保留当前对话的重要需求、实现状态、故障证据和后续工作，不是逐字聊天记录。**已验证结果、历史报告、待验证判断必须分别理解。**

## 1. 最新任务与当前状态

用户最新要求：只写好 `HANDOFF.md`，保留当前对话信息。用户已明确表示脚本修复不属于当前任务；不要继续修脚本、修改训练代码或启动测试。本文记录历史修复和故障仅用于交接，不代表获得继续处理这些问题的授权。

此前的紧急任务是修复服务器正式训练启动时错误依赖 `tests/configs` 的问题。这一修复现已完成 Windows 修改、WSL 同步和 CPU 回归测试；尚未部署到用户的正式服务器。

更早的 GRPO 本地端到端测试仍未完成：最后一次诊断确定程序停在首次训练模型到 vLLM 的权重同步，生成请求尚未发出。**根因未确认，问题未修复，不能宣称 GRPO 已跑通。**

用户曾明确要求停止工作并释放显卡。已停止测试，并确认当时没有训练、Ray、vLLM 残留进程；显存从约 41 GB 降到约 4.7 GB，剩余进程为 Windows 桌面应用。之后只有只读检查和 CPU 启动脚本测试。后续不要自动重启 GPU 长测试。

## 2. 工作目录和环境

| 用途 | 路径或环境 |
| --- | --- |
| 当前 Windows 正式仓库 | `C:\Users\qqian\Desktop\R^2OPL` |
| 原项目，仅供参考 | `C:\Users\qqian\Desktop\OPD_Forge` |
| WSL 当前测试仓库 | `/home/qqh/R2OPL` |
| WSL 原项目 | `/home/qqh/OPD_Forge` |
| WSL Python | `/home/qqh/miniconda3/envs/verl/bin/python` |
| WSL Conda 环境 | `verl` |
| WSL 模型 | `/home/qqh/R2OPL/models` 复用 `/home/qqh/OPD_Forge/models`，不再复制 |
| 当前服务器项目路径（用户报错） | `/mnt/cephfs/linkinleeli/R2OPL` |
| 正式硬件 | 8 × H100 |
| 本地硬件 | 单张 RTX 4090，约 48 GB 显存 |
| 历史下载代理 | `localhost:7897` |

注意：工具默认 cwd 仍可能是 `C:\Users\qqian\Desktop\OPD_Forge`。每次操作显式指定 R²OPL 路径，避免改错项目。历史 DAPO 记录里还出现 `/home/qqh/qqh/OPD_Forge`，这是历史路径，不能当成当前测试目录。

工作流：先修改 Windows R²OPL 正式目录，再同步到 WSL R2OPL，在 WSL 的 `verl` 环境验证。Windows 是最终提交代码的来源。不得删除、精简或改写 OPD_Forge 的内容。

## 3. 用户的强制约束

- 所有正式训练统一通过 `bash scripts/start_train.sh <配置文件>` 启动。
- 正式配置面向 8 卡：PG-OPD 是 4 Student + 4 Teacher；GRPO 是 8 卡 Student。
- 不能把正式分布式训练改成本地单卡串行流程。
- 正式 8 卡不做参数或优化器 offload。PG-OPD 已有 chunk 机制，应保留。
- Student、Teacher 的 `enable_sleep_mode` 都必须为 `false`；也要阻止框架通过其他开关调用 sleep/wake。无 Teacher 的算法不能访问 Teacher 导致报错。
- 只支持 no-thinking，作为隐含默认值，不在 run name、配置名或提示词命令中反复强调。
- 默认提示词是 `EXPLICIT_STEP_PROMPT`，配置键当前为 `prompt_template: explicit_step_prompt`。用户另有 `CROSS_DOMAIN_PROMPT`；除非显式覆盖，否则用前者。
- 全算法默认 64 questions/batch，每题 4 rollouts，共 256 条轨迹。
- W&B project 固定为 `R^2OPL`，链接：<https://wandb.ai/njuqqh/R%5E2OPL>。每种算法设置对应 group。
- 共享且被显式继承的基础配置放 `configs/` 根目录；具体实验放 `configs/<algorithm>/`。
- **测试配置最终要求放 `tests/configs/`。** 用户曾短暂要求放 `configs/`，后来明确纠正；以最终要求为准。
- 所有本地测试专用代码、脚本、fixture、工具都放 `tests/`。
- 本地 rollout 必须用 vLLM，并尽量继承服务器 vLLM 设置；不能用 HF 自回归替代后声称验证了服务器路径。
- 模型放不下时 Student、Teacher 分阶段加载并释放；本地 Gemma 31B Teacher 可用 E4B 替代测试。后来用户要求跳过 Gemma；当前 GRPO 只需验证 Qwen3-1.7B。
- 对过长 prompt 默认右截断，保留开头、截掉末尾。不得因题目过长终止训练/评测；不得硬编码必须有某个样本总数（例如 1128）。
- 优先使用 rollout 原始 token IDs、原始 token index，不能 decode 后重新拼接 tokenize 来重建位置。牢记 `tokenizer(A+B) != tokenizer(A)+tokenizer(B)`。

## 4. 数据集历史与当前约定

当前正式数据文件：

- `data/Math–Science–Logic-32K.json`：训练集。
- `data/MSL_Eval.json`：多个评测子集的容器。

训练集历史要求：

- Math：从 DAPO-17K 仅英文题目随机选 10667 条。
- Science：来自 Nemotron-Science 的 `MCQ.jsonl`，10667 条；如有可靠难度标注优先较难题。
- Logic：来自 `LogiQA2.0_train_fol.jsonl`，10666 条。
- 总共 32000 条，字段为 `id`、`question`、`answer`、`type`；`id` 是从 `00001` 到 `32000` 的五位字符串。
- `type` 仅允许 `Math`、`Science`、`Logic`。
- 选择题的选项放进 question，answer 是选项标签；开放题 answer 是答案。
- 清理 question 开头冗余的 `Question:\n`、`Passage:\n` 等前缀。
- JSON 编码要求：UTF-8、`indent=4`、`ensure_ascii=False`。
- 用户问过 solution 覆盖率，随后取消，不是当前待办。

正式评测默认 8 个子集，当前配置名：

1. `AMC23`
2. `AIME24`
3. `AIME25`
4. `MATH-500`
5. `GPQA-D`
6. `SciBench`
7. `LogicBench`
8. `LogiQA2.0`

超过 200 条的评测子集随机缩减到 200。Science、Logic 应排除纯数学推理部分。历史上曾加入 AIME26，用户后来要求从默认评测移除；不能再次自动加入。

`data.val_files` 指定物理 JSON 文件，`data.val_datasets` 明确选择容器内哪些子集参与评测。不能把容器内所有内容无条件全部评测。

这些数据加工要求来自历史对话，本次交接没有重新逐条审计数据文件。

## 5. 答案验证与 step 划分

- 相关工具：`utils/answer_verifier.py`、`utils/conservative_step_splitter.py`，具体实现状态以当前文件为准。
- 答案验证的相关逻辑应聚合到单个 Python 文件，不依赖用户自己的其他验证模块。允许外部 `math_verify`。
- 要支持 Qwen3 和用户口中的 Gemini/Gemma、Math/Science/Logic 三类回答。
- MCQ 接受 `\boxed{\text{A}}`、其他 LaTeX 变体、选项字母加内容、`Choice D` 等明确选项标记；不接受只有选项内容、没有选项标签的回答。
- step 划分应尽量少依赖外部库，未来能作为独立工具使用。
- 原则：不确定是否完整时优先合并，避免把一个完整推理操作拆成多个残缺片段。
- 曾使用 `data/math_science_logic_600_rollouts`：两个模型 × 三种题型 × 每类 100 条，共 600 条。用户不满意 Gemma 的 hard-only 划分过粗，随后优先处理答案验证。
- 历史名称混用 Gemini/Gemma；实际配置使用 `student_model_family: gemini4`，模型文件名是 `gemma-4-...`。不要无依据批量改名。

## 6. 算法和公共运行代码

主要文件：

- `algorithms/pg_opd.py`
- `algorithms/grpo.py`
- `utils/opd_runtime.py`
- `utils/training_entrypoint.py`
- `utils/ersr.py`
- `scripts/start_train.sh`
- 内置 VERL：`verl/verl/...`

公共逻辑已拆为：

- `BaseR2OPLTrainer`：公共验证、Avg@K、W&B 等。
- `BasePGOPDTrainer`：错误轨迹 Teacher replacement ERSR。
- `GRPOTrainer`：正确轨迹 Student action ERSR。
- 入口按 `algorithm.name` 选择算法。
- GRPO 不创建 Teacher/Reference/Critic；ERSR Teacher callback 可为空。

用户关心解耦是否影响 PG-OPD：已有 CPU 契约测试；没有完成正式 8 卡 GPU 验证，不能承诺所有分布式运行已验证。

### PG-OPD

基础配置：`configs/pg_opd.yaml`，继承 `configs/base.yaml`。

具体配置：

- `configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_1p7b_len8k_100steps.yaml`
- `configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_4b_len8k_100steps.yaml`
- `configs/pg_opd/pg_opd_gemma4_31b_it_to_e2b_it_len8k_100steps.yaml`

100 steps，训练生成上限 8192，评测生成上限 20480；当前 max prompt 2048、max model length 22528。训练 temperature=1、top_p=1、top_k=-1；评测 temperature=0.6、top_p=0.95、top_k=-1。reverse-KL、REINFORCE，保留蒸馏 chunk。

当前服务器模型路径：

| 模型用途 | 路径 |
| --- | --- |
| Qwen3-1.7B Student | `/mnt/cephfs/LLM_MODEL_HUB/qwen/Qwen3-1.7B` |
| Qwen3-4B Student | `/mnt/cephfs/LLM_MODEL_HUB/qwen/Qwen3-4B` |
| 两个 Qwen PG-OPD 配置的 Teacher | `/mnt/cephfs/LLM_MODEL_HUB/qwen/Qwen3-4B` |
| Gemma Student | `/mnt/cephfs/LLM_MODEL_HUB/gemma/gemma-4-E2B-it` |
| Gemma Teacher | `/mnt/cephfs/LLM_MODEL_HUB/gemma/gemma-4-31B` |

注意：Qwen Teacher 文件名/run name 写 Instruct-2507，但配置实际指向用户提供的 `Qwen3-4B` 目录。未检查服务器目录内实际模型版本，不能仅凭目录名断言正确或错误。不要擅自编造不存在的服务器 Instruct 路径。

### GRPO

基础配置：`configs/grpo.yaml`。

具体配置：

- `configs/grpo/grpo_qwen3_1p7b_len12k_500steps.yaml`
- `configs/grpo/grpo_qwen3_4b_len12k_500steps.yaml`
- `configs/grpo/grpo_gemma4_e2b_it_len12k_500steps.yaml`

要求：500 steps，训练生成上限 12288，评测 20480；clip ratio、low、high 都是 0.2；`adv_estimator: grpo`、`norm_adv_by_std_in_grpo: true`、`loss_agg_mode: seq-mean-token-mean`。每条轨迹按有效 response tokens 数归一化。无 actor KL、reward KL、Teacher、Reference、Critic。W&B group `GRPO`。

## 7. ERSR 评测定义

早期称 ESER，后来用户明确纠正为 ERSR。

对 Student response 划分 steps。在 step k 之前的共同前缀 h(k-1) 上：

- V_before：Student 从前缀重新决定并续写的最终正确率。
- V_student：保留原 Student step k 后，由 Student 续写的最终正确率。
- V_teacher：Teacher 给出替换 step k 后，由 Student 续写的最终正确率。
- Student action advantage = V_student − V_before。
- Teacher replacement advantage = V_teacher − V_before。

每个 V 用 `mc_k` 条 Student-policy continuation 的平均奖励估计。Teacher 只提供替换步骤，后续仍由 Student 生成。

选择规则：

- OPD-only：仅错误轨迹，Teacher replacement。
- RL-only（GRPO）：仅正确轨迹，Student action。
- 混合算法：正确轨迹 Student action，错误轨迹 Teacher replacement。
- ERSR datasets 必须是 `val_datasets` 的子集。
- 不能选择最后一个 step。
- 某数据集无合格轨迹/step 时输出 0，不得报错中止。
- Teacher 直接给最终答案、Student continuation 为空，也允许有效评分，不能导致训练终止。
- 保留原始 token IDs 和 step 边界，不能用重新 tokenize 的字符串重建对齐。

正式默认：AMC23、AIME24、SciBench、LogicBench；每个数据集最多 500 个选中 steps；`mc_k=2`。每个 step 需要 baseline 和 action 两组 continuation，因此最多 500×2×2=2000 条 Student continuation/数据集，不含原始验证轨迹和 Teacher 替换生成。合格 steps 少则实际更少。

W&B：保留原来每个子集的 chart；另有集中展示所有子集 Avg@16 的记录。ERSR 各数据集/动作类型也集中记录，不能拆成一堆独立 chart。

## 8. 测试状态与必须纠正的历史表述

### CPU 历史结果

前一模型的交接报告曾记录：全量 `111 passed`，之后 GRPO + PG-OPD 针对性 `42 passed`。本次没有重新复现全量结果，不能将这些数字视作最新代码的重新验证。

### Qwen PG-OPD 已有分阶段测试报告

WSL 文件：`/home/qqh/R2OPL/tests/artifacts/pg_opd_qwen_vllm_smoke/report.json`。

- status=passed。
- 4 questions × 4 responses，2 个优化 batch。
- AMC23 两题 Avg@4=0.125。
- Teacher replacement ERSR 实际选中 1 个 case，MC=1，advantage=0。
- 全流程约 251.9 秒；Student rollout+validation 阶段约 93.5 秒。

**验证范围限制很重要：** `tests/run_pg_opd_single_gpu_smoke.py` 先用独立 vLLM 进程产生两批 rollout/验证数据，再依次做 Teacher、ERSR，最后加载训练 Student，用预存轨迹做两次 SGD optimizer step。没有经过正式训练 Actor→vLLM 热同步，也没有验证第一批更新后将新权重同步再生成第二批。不能把这个报告当作完整正式训练循环通过的证据。

### GRPO 当前 smoke

配置：`tests/configs/grpo/grpo_qwen3_1p7b_single_gpu_smoke.yaml`。

- 只测 Qwen3-1.7B。
- 用户同意缩到每 batch 2 questions × 4 responses，共 8 条。
- 需要 2 个真正训练 batch。
- 训练/验证生成上限 4096。
- `val_before_train=false`，第二步后验证 AMC23 两题 Avg@4，并做最小 Student-action ERSR。
- 单卡训练和 vLLM 同时驻留，关闭 offload、sleep。
- 当前未完成；不能报告训练、验证、ERSR 已通过。

## 9. GRPO 卡住：已确认事实

失败/中止运行日志：

- `/home/qqh/R2OPL/tests/artifacts/grpo_qwen3_1p7b_single_gpu_smoke/run.log`
- `/home/qqh/R2OPL/tests/artifacts/grpo_qwen3_1p7b_single_gpu_smoke/diag128.log`

最后一次短测只把 train generation 改为 128 tokens，仍卡住。运行期间 Ray task 状态明确显示：

- `WorkerDict.actor_rollout_update_weights`：RUNNING。
- `vLLMHttpServer.collective_rpc`：RUNNING。
- 没有开始 `AgentLoopWorkerTQ.generate_sequences` 或 `vLLMHttpServer.generate`。

`main_ppo_sync.py` 的 fit 在进入训练进度循环前调用 `checkpoint_manager.update_weights()`。所以这次等待发生在首次 Actor→vLLM 权重同步，**不是 8 条 rollout 生成太慢**。

之前助手依据 GPU 利用率和理论 continuous batching，把等待解释成低吞吐，又让用户等了很久；该解释没有证据，已经向用户纠正。不要继续沿用，也不要把 128-token 诊断或中止运行算成成功。

根因仍未确定：发送、IPC 握手、接收、加载参数中的哪一步未返回，没有捕获到具体阻塞栈。CUDA IPC、ZMQ 通信、vLLM collective RPC 是排查对象，不能写成已确认根因。

## 10. OPD_Forge 与 R²OPL 对照发现

### 历史 4B DAPO 基准真实存在

文件：`/home/qqh/OPD_Forge/data/DAPO-17k-Qwen3-4B-NoThinking-Rollouts-Max8192/summary.json` 及 `manifest.json`。

- 17255 条、31206665 个 response tokens。
- 平均长度 1808.56 tokens。
- batch=16、max_new_tokens=8192、max_model_len=9730。
- temperature=0.6、top_p=0.95、top_k=-1。
- `gpu_memory_utilization=0.4`、`max_num_batched_tokens=1024`、`enforce_eager=true`。
- 用户提供历史速度：约 263 tokens/s，6.9 秒/条、8.8 条/分钟、总计约 32.7 小时。

对应脚本 `tests/generate_qwen3_1_7b_rollouts_with_logprobs.py` 直接 `vllm.LLM(model=...)` 加载文件权重，调用 `llm.generate(...)`，没有 Actor→vLLM 动态权重传输。因此该基准证明本地 vLLM 正常推理可以很快，但未覆盖此次卡住的环节。

### 正式 rollout 调度

两边正式训练都使用 async rollout、AgentLoopWorkerTQ 和 LLMServerManager。一个 question 的 n 条 response 通过 asyncio task 并发提交，再由同一个 vLLM server 调度。用户已接受该架构本身与 OPD_Forge 一致。

不能把“理论支持 concurrent batching”当成某次请求确实进入调度器的证据；此次短测尚未到请求生成。

### 权重同步核心文件相同

Windows 两仓库哈希对比相同；WSL 两仓库以下 6 个实际文件的 SHA256 也相同：

- `verl/verl/workers/engine_workers.py`
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
- `verl/verl/workers/rollout/vllm_rollout/utils.py`
- `verl/verl/utils/device.py`

当前继承的正式设置包含 `load_format: dummy`、checkpoint backend `naive`、权重传输 bucket 2048 MiB。vLLM 需要首次同步才能使用训练模型的真实权重。不能仅改 `load_format=auto` 绕开后就宣称解决了每个 optimizer step 后的同步。

`is_support_ipc()` 对 CUDA 直接返回 True，没有在这里验证当前 WSL 的 IPC 能力；这是排查线索，不是已证实不兼容。

### OPD_Forge 也有未通过的正式入口本地测试

WSL 日志：`/home/qqh/OPD_Forge/tests/.local_pipeline_logs/r2opl_base_v2_local_pipeline_smoke.log`，2026-09-16。

该日志在引擎启动、wake weights、FSDP state dict 附近之后反复出现 `No available shared memory broadcast block found in 60 seconds`，最终 Raylet 被终止。不能把这份旧测试当作正式热同步链路曾经通过的基线。它与本次存在类似等待现象，但尚未证明根因相同。

### 已证实的配置传递缺陷（尚未修）

`verl/verl/workers/rollout/vllm_rollout/utils.py` 的 `build_cli_args_from_config()`（约第 701 行）把布尔 True 转成 flag，却直接丢弃 False。

旧 OPD_Forge 测试明确设置 `engine_kwargs.vllm.async_scheduling=false`；完整配置日志确实是 False，但 vLLM 启动日志显示 `Asynchronous scheduling is enabled.`。所以存在“配置写 false，却没有真正关闭默认开启选项”的具体缺陷，两个项目都有。

这与 `rollout.mode: async` 是不同层次，不能混淆。尚未证明修复该传递问题能解决权重同步阻塞。修复布尔参数时需识别 vLLM 对每个选项支持的否定 flag，不能一概对所有 False 拼接参数。

### 本地测试配置的其他差异

- 新 GRPO smoke 继承 8 个 AgentLoop workers、4 个 reward workers、8 个 dataloader workers，并渲染完整 32K 数据集；旧低资源正式入口 smoke 用 1/1/0 和 32 条训练子集。这会增加 CPU/RAM 和启动开销，尚无证据证明是同步卡住的根因。
- 新 GRPO 用 CUDA graphs，历史 DAPO 基准用 eager；前者不等于必然更慢。
- 正式项目按用户要求 sleep=false、offload=false，不能为尝试排错随意恢复旧默认。
- 短测虽然打开 `disable_log_stats=false`，Ray runtime 仍有 `VLLM_LOGGING_LEVEL=WARN`。要观察 INFO 级吞吐需要一起核对日志级别，不能承诺仅改一个开关就能看到统计。
- 权重传输使用阻塞 socket 等待，缺少明确的阶段日志/超时诊断，容易将卡死表现成无输出等待。

## 11. 服务器启动报错：本次已修复

用户执行：

```bash
bash scripts/start_train.sh configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_4b_len8k_100steps.yaml
```

错误：

```text
realpath: /mnt/cephfs/linkinleeli/R2OPL/tests/configs: No such file or directory
```

原因已确认：启动脚本在读取/判断用户给出的正式配置前，无条件执行：

```bash
test_config_root=$(realpath "$project_root/tests/configs")
```

同时 `set -e` 使该失败立即退出。`.gitignore` 明确忽略 `/tests/`，服务器没有本地测试目录本来就是正常的。问题完全在脚本，与用户明确指定的正式配置无关；不是 YAML 内容错误，也还没有进入模型加载。

已修改 Windows `scripts/start_train.sh`：

- 先将可选测试目录保存为路径字符串，不要求它存在。
- 读取并解析用户实际指定的配置文件。
- 仅当选中的配置不在正式 configs 内、且测试目录存在时，才 realpath 测试目录。
- 正式配置只使用正式 configs 和 VERL 的 Hydra 搜索路径。
- 保留本地 `tests/configs` 启动支持。

测试文件：

- 新增 `tests/test_start_train.py`：真实执行 Bash 启动脚本，用 fake Python 捕获 argv，完全不加载模型。
- 删除 `tests/test_grpo.py` 中只断言错误脚本文本存在的旧测试；以行为测试替代。

已将上述 3 个文件同步至 WSL `/home/qqh/R2OPL`。2026-09-19 22:11 左右验证：

```text
bash -n scripts/start_train.sh：通过
pytest tests/test_start_train.py：10 passed in 0.16s
```

覆盖：tests 完全不存在、tests 存在但 configs 不存在、tests/configs 存在；正式/本地配置的相对/绝对路径；项目路径含空格；Hydra 参数和 override 正确保留；指定文件不存在；配置在允许目录之外。

CPU 测试使用 `CUDA_VISIBLE_DEVICES=""`；fixture 位于 `tests/.launcher_regression_20260919_2211/`。该 pytest basetemp 已存在，不要复用可能触发清理的固定目录来存放其他数据。

**部署状态：没有服务器连接，未替用户更新服务器脚本。** 用户需把修复后的 Windows `scripts/start_train.sh` 同步到服务器再运行原命令。仅需更新启动脚本解决这次报错，不必创建虚假的 tests 目录。这个修复不代表权重同步或 8 卡训练已经通过。

## 12. 当前文件修改和提交注意

创建本文前，Windows `git status --short` 显示：

```text
 M configs/base.yaml
 M scripts/start_train.sh
```

- `configs/base.yaml` 是此前按用户要求关闭正式 param/optimizer offload 的未提交变更，必须保留。
- `scripts/start_train.sh` 是本次启动修复。
- `tests/test_start_train.py` 和 `tests/test_grpo.py` 的修改位于被 `.gitignore` 忽略的 tests 下，不会出现在普通 git status；不要误以为它们不存在。
- 本文为新增 `HANDOFF.md`。
- 没有 commit、push 或服务器部署；不要声称已提交。
- OPD_Forge 本轮只读，没有删除或修改。

## 13. 后续优先级与沟通原则

1. 当前只交付本交接文档，不继续处理脚本修复、服务器同步或其他代码工作。下列技术事项仅供用户未来明确要求继续时参考，不是当前执行计划。
2. 若继续处理权重同步，先定位发送和接收各自的阻塞点，检查 endpoint/job/rank 一致性以及 CUDA IPC 握手，不要重新盲跑 4K。
3. 单独处理已经证实的布尔 false 参数丢失；以 CPU 参数解析回归保证设置真正生效，不能只检查 YAML 字面值。
4. 只有用户恢复 GPU 调试后，再做有时间上限的最小权重同步检查和短生成。成功后才做两批真实 GRPO、AMC23 两题 Avg@4、最小 Student-action ERSR。
5. 验证要包含训练更新后的第二次权重同步和第二批生成；预先生成两批再离线优化不能替代在线循环测试。
6. 对正式 8 卡必须明确仍有哪些分布式行为未在本地验证。

用户因低效测试、错误进度判断和基础启动错误已经非常不满。回答应简短、准确、先给证据；不反复道歉，不用未经测量的速度/预计耗时安抚。不要承诺“几十秒就知道”后又让黑盒等待。当前没有已确认的 GPU rollout 吞吐下降根因，不要把猜测写成结论。
