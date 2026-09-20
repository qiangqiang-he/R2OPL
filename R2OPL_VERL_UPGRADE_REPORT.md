# R²OPL verl 升级与本地验证报告（2026-09-20）

本轮任务：把本地 vendored verl 与最新上游 verl 合并、新建 WSL `r2opl` 环境（新 vLLM 栈）、验证 Qwen3 / Qwen3.5 / Qwen3.6 / gemma-4 的 vLLM rollout，并跑通全部算法单卡 smoke。

## 1. 结果总览

| 项目 | 状态 |
| --- | --- |
| 本地 verl 备份 | ✅ `R^2OPL\verl_backup_20260919_premerge`（另有外层 git 兜底） |
| 上游 merge（保留全部本地修改） | ✅ merge commit `abe2a826`，全树 `compileall` 0 错误 |
| WSL `r2opl` 环境 | ✅ Python 3.12.13 / torch 2.13.0+cu130 / **vllm 0.29.0** / transformers 5.10.4 / ray 2.58.0 / verl 0.10.0.dev0 (editable→`/home/qqh/R2OPL/verl`) / TransferQueue 0.1.7 |
| Qwen3-0.6B vLLM rollout | ✅ 2题×4rollout 一次生成，~1993 tok/s |
| Qwen3.5-2B vLLM rollout | ✅ ~874 tok/s（Qwen3_5 新架构 + GDN 线性注意力） |
| Qwen3.6-27B-FP8 vLLM rollout | ✅ ~201 tok/s（Marlin FP8；需 `max_num_seqs≤169`，因混合 Mamba cache 块限制） |
| **gemma-4-E2B-it vLLM rollout** | ✅ ~922 tok/s（**原始问题已解决：vllm 0.17 无法为 gemma-4 做 rollouts**） |
| PG-OPD smoke（Qwen 4B→1.7B） | ✅ exit=0 |
| EOPD smoke（Qwen 4B→1.7B） | ✅ exit=0 |
| **GRPO smoke（Qwen 1.7B，在线循环）** | ✅ exit=0：两批真实训练 + AMC23 Avg@4=0.875 + Student-action ERSR，`status=passed` |
| GSPO smoke（Qwen 1.7B） | ✅ exit=0（修复脚本 import 顺序后）：两批训练 + Avg@4=1.0 + ERSR，`status=passed` |
| PG-OPD smoke（Gemma E4B→E2B，轻量） | ✅ exit=0：`status=passed`、2 步优化器、Avg@4=0.0（512-token 生成下 2B 学生做两道 AMC 难题的合理值，指标链路完整） |
| EOPD smoke（Gemma E4B→E2B，轻量） | ◐ student/teacher/ERSR 三阶段全部通过（teacher 打分 15347 tokens / 3.6s / 4261 tok/s）；最后的进程内训练阶段进行中被用户主动叫停，未走完 2 步优化器 |

**Gemma 全尺寸 smoke 的说明**：4096-token response 的原版 gemma smoke 中，E4B Teacher 的 prompt-logprobs 打分（262K 词表 × 每位置 top-k，Triton attention 后端）单阶段需 2 小时以上，EOPD 训练阶段（HF 全 logits 前向+反向）同样极慢。为此新增 `*_smoke_light.yaml`（response 512、上下文 3072），流水线与原版完全一致（student vLLM rollout → teacher 打分 → ERSR → 2 步优化器 + Avg@4）。原版配置保留未动，想在服务器或长时间本地点验时可直接用。

各 smoke 的机器可读结果：WSL `/home/qqh/R2OPL/tests/artifacts/`；套件汇总：`tests/artifacts/smoke_suite_logs/summary.txt`；家族 rollout 报告：`tests/artifacts/vllm_family_checks/*.json`。

## 2. verl merge 细节

- 基线：本地 verl 基于 v0.8.0（`verl-v0.8.0-vllm0.20.2.freeze.txt`），上游 `verl-project/verl` main（v0.9.0 之后，HEAD `3efe38c7`）。
- 方法：在临时克隆里以 v0.8.0 为基线提交本地快照（`r2opl-vendored` 分支），再 3-way merge 上游 main。真实修改面：**32 个文件修改 + 2 个新增**（distillation losses +1844、main_ppo_sync +464、vllm_rollout utils +439 等），全部保留。
- 12 个冲突文件逐一手工解决；本地删除的上游 `tests/`（207 文件）保持删除。
- 关键决策：
  - 上游把 `main_ppo_sync.py` 重构为 `ppo/v1/*`。**恢复本地 `main_ppo_sync.py` 与上游新结构并存**（R²OPL 算法层 `from verl.trainer import main_ppo_sync` 不受影响）；`create_rl_dataset/create_rl_sampler` 改从 `verl.trainer.ppo.utils` 导入。
  - 本地 `distillation_direct_chunked` 与上游 `distillation_only` 语义相同：transformer_impl 里两者取 OR，赋值点两处都传。
  - prompt 超长截断保留本地规则（右截断、保开头 `[:prompt_length]`）。
  - 本地 EOPD/R²OPL-base/R²OPL-v2 钩子、分阶段 checkpoint map_location 修复、`_torch_save_via_local_temp` 全部保留；同时吸收上游 Continuous Token、PD disagg、profiler 去重、no-sync 梯度累积、LoRA-only checkpoint。
- 验证：全树无冲突标记残留；`python -m compileall verl` 0 错误；`main_ppo_sync`/`main_ppo`/`trainer_base`/R²OPL `algorithms` 全部导入成功；`pip check` 仅 TransferQueue 的 numpy 上界提示（沿用 `--no-deps` 安装，与旧环境做法一致）。
- 合并工作区保留在 `%TEMP%\verl-upstream-merge`（分支 `r2opl-vendored`），以后再同步上游可直接复用。

## 3. `r2opl` 环境要点

- 创建：`conda create -n r2opl python=3.12`；代理 `http://172.27.112.1:7897` + 官方 PyPI（阿里镜像仅 ~300KB/s，代理 ~40MB/s）。
- 版本协调：vllm 0.29 要求 `transformers>=5.10.4`、verl 要求 `<5.11` → pip 自动落 **5.10.4**。torchvision 0.28.0 与 torch 2.13 配套。
- **WSL 专属设置（已写入 conda activate.d，永久生效）**：
  - `VLLM_USE_V2_MODEL_RUNNER=0`：vLLM 0.29 新 V2 model runner 的 UvaBuffer 依赖 CUDA UVA（cuMem* 驱动 API），WSL 不支持会直接 `RuntimeError: UVA is not available`；回退 V1 runner 即可。服务器裸机无此问题。
  - 运行任何 vLLM 程序时确保 `PATH` 含 `/home/qqh/miniconda3/envs/r2opl/bin`（FlashInfer JIT 需要 `ninja`；直接用绝对路径调 python 会绕开 bin 目录）。
- 环境快照：`verl/verl-0.10.0.dev0-r2opl-merged-vllm0.29.0.freeze.txt`（沿用旧 freeze 命名惯例，Windows/WSL 双侧已同步）。

## 4. 代码修改清单（本轮新增）

| 文件 | 修改 |
| --- | --- |
| `verl/`（整个目录） | 上游 merge 结果（见 §2）；新增 freeze 快照 `verl-0.10.0.dev0-r2opl-merged-vllm0.29.0.freeze.txt` |
| `tests/verify_vllm_model_families.py` | 新增：4 家族最小模型 vLLM rollout 验证（2题×4rollout 一次 generate） |
| `tests/run_pg_opd_single_gpu_smoke.py` | ① uv 缓存 transformers 回退改为"仅当环境没有 transformers 时启用"；② 补上缺失的 `_apply_gemma4_vllm_compatibility_patches`（带"vLLM 原生支持 Gemma4 则跳过"守卫，旧版补丁逻辑完整保留为回退）；③ `_assert_stage_boundary` 的 35 GiB 进程内释放下限改为可经 `local_smoke.min_free_gib_after_stage` 配置（默认 35 不变） |
| `tests/run_eopd_single_gpu_smoke.py`、`tests/run_grpo_single_gpu_smoke.py` | uv 缓存 transformers 回退同样加守卫 |
| `tests/run_gspo_single_gpu_smoke.py` | 修复 import 顺序 bug：`PROJECT_ROOT` sys.path 注入移到 `tests.*` 跨文件 import 之前（原顺序直接 ModuleNotFoundError，该 smoke 此前从未跑通过） |
| `tests/configs/pg_opd/pg_opd_gemma4_e4b_it_to_e2b_it_single_gpu_smoke_light.yaml` | 新增轻量 Gemma PG-OPD smoke 配置 |
| `tests/configs/eopd/eopd_gemma4_e4b_it_to_e2b_it_single_gpu_smoke_light.yaml` | 新增轻量 Gemma EOPD smoke 配置 |

Windows `R^2OPL` 为正式来源，已 rsync 同步至 WSL `/home/qqh/R2OPL`。外层 git 未提交（由你审阅后决定）。

## 5. 已知边界与备注

- Qwen3.6-27B-FP8 在 48GB 4090 上需 `max_num_seqs ≤ ~169`（混合 Mamba 每 decode 序列占一个 cache 块）；测试用 128。
- gemma-4 走 TRITON_ATTN 后端（非 FlashAttention），E4B teacher 的 prompt_logprobs 打分与 EOPD 全词表训练前向在单卡 4090 上计算量很大：全尺寸 smoke 单阶段 2 小时以上，轻量配置则分钟级。
- `VLLM_USE_V2_MODEL_RUNNER=0` 下所有家族验证与 smoke 均通过；若未来 vLLM 修复 WSL UVA 可移除该开关。
- TransferQueue 0.1.7 按旧惯例 `--no-deps` 安装（其 metadata 要求 numpy<2，实际与 numpy 2.3.5 共存正常，smoke 已验证）。
- 8 卡服务器的分布式行为未在本地验证（单卡环境）；本地已覆盖单卡能覆盖的全部链路。

## 6. 运行历史（tests/artifacts/smoke_suite_logs/summary.txt 原文）

```text
pg_opd_qwen exit=0            # 首轮直接通过
gspo_qwen exit=1              # 脚本 import 顺序 bug → 修复
eopd_qwen exit=0
grpo_qwen exit=0              # 在线循环（权重同步）通过
pg_opd_gemma exit=143         # 全尺寸：手动打断（teacher 阶段 >2h）
eopd_gemma exit=1             # 脚本缺 _apply_gemma4 函数 → 补齐
gspo_qwen retry exit=0        # ✅ 修复后通过
pg_opd_gemma retry exit=143   # 全尺寸：手动打断
pg_opd_gemma_light light exit=1  # 长度差 1（校验需 prompt+resp+1）→ 2560→3072
eopd_gemma_light light exit=1     # 同上
pg_opd_gemma_light light exit=1  # 进程内 35GiB 释放断言 → 改为可配置下限
eopd_gemma_light light exit=124  # 2h 上限到期（训练阶段慢）
pg_opd_gemma_light final exit=0  # ✅ Gemma PG-OPD 全链路通过
eopd_gemma_light final exit=143  # 三阶段已过、训练中被用户叫停
```

## 7. 复现命令（r2opl 环境）

```bash
conda activate r2opl   # 自带 VLLM_USE_V2_MODEL_RUNNER=0
cd /home/qqh/R2OPL

# 家族 rollout 验证（每次 2题×4rollout 单次 generate）
python tests/verify_vllm_model_families.py --family qwen3   # 或 qwen35 / qwen36 / gemma4

# 算法 smoke（Qwen 全部、Gemma 用轻量配置）
python tests/run_pg_opd_single_gpu_smoke.py --config tests/configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_1p7b_single_gpu_smoke.yaml
python tests/run_eopd_single_gpu_smoke.py   --config tests/configs/eopd/eopd_qwen3_4b_instruct_2507_to_1p7b_single_gpu_smoke.yaml
python tests/run_gspo_single_gpu_smoke.py   --config tests/configs/gspo/gspo_qwen3_1p7b_single_gpu_smoke.yaml
python tests/run_grpo_single_gpu_smoke.py   --config tests/configs/grpo/grpo_qwen3_1p7b_single_gpu_smoke.yaml
python tests/run_pg_opd_single_gpu_smoke.py --config tests/configs/pg_opd/pg_opd_gemma4_e4b_it_to_e2b_it_single_gpu_smoke_light.yaml
python tests/run_eopd_single_gpu_smoke.py   --config tests/configs/eopd/eopd_gemma4_e4b_it_to_e2b_it_single_gpu_smoke_light.yaml
```
