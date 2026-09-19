# R²OPL

This migration currently contains one algorithm only: **PG-OPD**.

The matching VERL source tree is vendored under `verl/`. It is kept intact as
the distributed execution engine because pruning its worker, rollout, and
configuration internals would make the 4+4 GPU path fragile. R²OPL's own
algorithm registry and formal configuration still expose PG-OPD only.

PG-OPD trains on Student-sampled response tokens with the detached signed
advantage

```text
A = log π_teacher(a|s) - log π_student(a|s)
L = -E[stop_gradient(A) log π_student(a|s)]
```

## Data

- `data/Math–Science–Logic-32K.json`: 32,000 training questions.
- `data/MSL_Eval.json`: grouped evaluation questions, flattened in memory by
  the dataset adapter.

`data.val_files` identifies the physical grouped JSON container, while the
required `data.val_datasets` list explicitly selects the logical benchmarks
that participate in validation. Unknown, empty, or duplicate selections fail
instead of silently evaluating the wrong subset.

W&B retains the existing per-dataset validation metrics and additionally logs
`val-summary/Avg@K-all-datasets`, one multi-line chart containing only the
configured Avg@K history for every evaluation dataset (for example Avg@16 in
formal runs and Avg@4 in the single-GPU smoke configs).

PG-OPD also runs Expected Reasoning-Step Return (ERSR) during validation on
`AMC23`, `AIME24`, `SciBench`, and `LogicBench`.  It samples at most 500
non-terminal reasoning steps from incorrect trajectories per dataset.  For
each step, MC@2 Student continuations estimate the re-decide baseline and the
Teacher-replacement value.  All four mean replacement advantages share the
single W&B chart `val-summary/ERSR-all-datasets`.  Empty Student continuations
remain valid—for example, when the Teacher replacement already provides the
final answer.

No converted or duplicated dataset is required.
Overlong questions are retained: the adapter keeps the beginning of the
question, truncates its ending to the configured prompt budget, and then
renders the complete model-native assistant generation suffix. Training and
evaluation never drop a sample merely because its prompt is too long.

## Default training

The formal configuration targets one 8-GPU node:

- 4 GPUs for the Student actor/rollout pool.
- 4 GPUs for independent Teacher replicas.

Run it with:

```bash
bash scripts/start_train.sh configs/pg_opd.yaml
```

The three 8K-train / 20K-eval, 100-step experiment configurations are:

```bash
bash scripts/start_train.sh configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_1p7b_len8k_100steps.yaml
bash scripts/start_train.sh configs/pg_opd/pg_opd_qwen3_4b_instruct_2507_to_4b_len8k_100steps.yaml
bash scripts/start_train.sh configs/pg_opd/pg_opd_gemma4_31b_it_to_e2b_it_len8k_100steps.yaml
```

Formal configs use the server checkpoints under
`/mnt/cephfs/LLM_MODEL_HUB`. Student and optional Teacher vLLM engines always
run with sleep mode disabled; cache release is disabled as well so VERL never
calls the unsupported sleep/wake path.

All experiments use the W&B project
[`R^2OPL`](https://wandb.ai/njuqqh/R%5E2OPL). Run names describe only the
algorithm, model, and experiment-specific variables.

The launcher uses the bundled `verl/` tree by default. An explicit source
override remains available for engine development or comparison:

```bash
export R2OPL_VERL_ROOT=/path/to/verl
bash scripts/start_train.sh configs/pg_opd.yaml
```

Qwen3 is the default model family. Gemini 4 is enabled by overriding both
family fields and the two model paths together; mixed tokenizer families are
rejected because PG-OPD must score the Student's original sampled token IDs.

## Test boundary

Local migration tests are under `tests/` and are CPU-only. They validate the
prompt contract, both JSON layouts, boxed-answer rewards, signed advantages,
padding masks, Hydra composition, and the formal 4+4 resource-pool plan. Full
rollout, Teacher forward, backward, and distributed execution remain server
tests.
