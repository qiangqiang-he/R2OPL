# 本地正式入口回归

所有后续本地实验与测试统一使用 WSL 的 `r2opl-cu12`（vLLM 0.19.1）。在项目根目录运行：

```bash
conda activate r2opl-cu12
python tests/run_gsm8k_formal.py r2opl_base
python tests/run_gsm8k_formal.py r2opl_base --probe
python tests/run_gsm8k_formal.py pg_opd
python tests/run_gsm8k_formal.py eopd
python tests/run_gsm8k_formal.py opdvr
python tests/run_gsm8k_formal.py grpo
python tests/run_gsm8k_formal.py gspo
```

监控脚本实际启动 `bash scripts/start_train.sh tests/configs/gsm8k/<算法>.yaml`。
启动时校验环境及 vLLM 版本，结果文件记录关键依赖版本；旧 `r2opl` 的成功记录不代表此环境已通过。
它不预生成 rollout，不用离线样本代替正式训练。

- 模型：本地 `models/Qwen3-0.6B`；蒸馏算法 Teacher 和 Student 同模型、独立实例、同一张 GPU。
- 数据：`data/gsm8k_train.json` 取前 4 题，`data/gsm8k_test.json` 取前 2 题。源 JSON 无需修改。
- 每批 2 题，每题 4 条 rollout；训练 2 步后评测和保存模型/训练状态（不保存优化器大张量）。
- Student 生成总长度上限 1024：prompt 256，response 768；R²OPL Student 引擎容量 1152，保留追加自探针的空间。`--probe` 是额外的 96-token 截断覆盖测试。
- 训练和评测均 temperature=1.0、top_p=1.0、top_k=-1。
- 本地使用 SDPA、每个 microbatch 一条序列，关闭 remove_padding；FSDP、反向传播和在线权重同步仍使用正式代码。
- WSL 使用共享内存传权重；Teacher 显式共用 actor 资源池。正式 8 卡配置的默认行为不变。
- 每 2 秒记录显存，低于 4GiB 剩余即停止，为桌面预留至少 3GiB 的安全余量。超时 180 秒无 rollout 进展则停止排查。

日志、显存记录、退出状态和训练产物位于 `tests/artifacts/gsm8k/<算法>_gsm8k_2steps/`，旧尝试保存在 `attempts/`。
仅启动脚本或配置检查成功不算通过；必须完成两步训练、权重更新、评测和检查点保存。

全部运行后执行 `python tests/summarize_gsm8k_formal.py` 核对生成数量、每题 4 条、非空文本、权重更新与检查点，汇总写入 `tests/artifacts/gsm8k/summary.json`。还会逐条重新判题，检查依赖奖励的算法训练分数及全部评测分数；R²OPL 另核验探针字段、RL/OPD 数量、有效奖励和混合正误组的非零 RL 梯度。任何算法未完成或核验失败时返回非零状态。

合并兼容性回归使用 `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:./verl python -m pytest tests -q`。EOPD 的 dense SDPA 分块损失另有数值和梯度对照测试 `tests/test_dense_chunked_distillation.py`，覆盖不同长度及真实 Teacher entropy 列向量格式。

发布验证说明见 `R2OPL_BASE_RUNTIME_VALIDATION.md` 和 `PG_EOPD_RUNTIME_VALIDATION.md`。本地模型、GSM8K 数据与大检查点不上传 GitHub。其余算法的测试状态以本地汇总记录为准。
