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
bash scripts/start_train.sh configs/pg_opd/pg_opd_gemma4_12b_it_to_e2b_it_len8k_100steps.yaml
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

## Evaluation

`bash scripts/start_eval.sh configs/eval/<config>.yaml` runs standalone
vLLM evaluations that are independent of the training stack.  Every config
under `configs/eval/` is fully self-contained (there is no shared
`eval_base.yaml`).  Two model sources are supported: `training_output`
discovers every `global_step_*` checkpoint of a VERL run (merging FSDP actor
shards on demand) and `model_source: hf_models` evaluates explicit local
model directories.  On the formal 8-GPU server each GPU loads one complete
model replica and dynamically takes question batches of up to 256 rollouts
per generate call by default; an optional `ersr` block adds Expected
Reasoning-Step Return with the same seed arithmetic as training-time ERSR.
`prompt_template` selects `explicit_step_prompt` (default) or
`cross_domain_prompt` from `utils/prompts.py`.  Results land under
`eval_results/` as JSON (Avg@n, Pass@n, mean_length, truncation_rate at every
configured token limit, plus per-sample Avg@n and the ERSR report).
`--validate-only` checks the whole contract on CPU.

SciBench uses numerical grading with 5% relative tolerance, including ERSR
continuations; other datasets retain exact math/choice grading. SciBench
questions must specify the required unit; the scorer does not convert units.
Training rewards are unchanged.

ERSR defaults to at most 200 non-final reasoning steps **per dataset and per
action** (`max_steps_per_dataset`). `teacher_replace` requires `ersr.teacher`;
mixed-family model lists can instead use `ersr.teachers.qwen3` and
`ersr.teachers.gemini4`, each with `name` and `path`. Teacher and Student
token-to-ID vocabularies are checked before generation.

Checkpoint evaluation finishes and saves all base metrics first, runs one
Teacher phase for all selected checkpoints, releases the Teacher, then runs
Student MC continuations checkpoint by checkpoint. All phases use every GPU
in `runtime.gpus` (set `[0]` for local testing). FSDP actor shards are merged
once on CPU; temporary merged weights remain on disk until their Student
phase finishes and are removed on success or failure. Direct HF model lists
complete these phases separately for each model. Raw rollouts are saved when
`report.save_rollouts: true`; checkpoint results are saved after each phase.

## Test boundary

Local tests are under `tests/`, including CPU regressions and single-GPU
evaluation smoke tests through `start_eval.sh`. They cover model lists,
FSDP checkpoint merging, staged ERSR, dataset grading, and memory headroom.
The evaluation smoke configs are `tests/configs/eval/phase_models_single_gpu.yaml`
and `phase_checkpoints_single_gpu.yaml`. Full eight-GPU execution still
requires the server; local checkpoint fixtures use eight CPU DTensor ranks.
