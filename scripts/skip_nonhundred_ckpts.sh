#!/usr/bin/env bash
# Rename non-hundred VERL checkpoint directories so the ERSR evaluator ignores
# them. The evaluator discovers only names matching ^global_step_[0-9]+$.
#
# Default: print the planned moves only.
# Apply:   APPLY=1 bash scripts/skip_nonhundred_ckpts.sh <training_output>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: APPLY=1 bash scripts/skip_nonhundred_ckpts.sh <training_output>" >&2
  exit 2
fi

run_dir=$1
if [[ ! -d "$run_dir" ]]; then
  echo "Training output directory does not exist: $run_dir" >&2
  exit 2
fi
run_dir=$(realpath "$run_dir")

apply=${APPLY:-0}
if [[ "$apply" != 0 && "$apply" != 1 ]]; then
  echo "APPLY must be 0 (preview) or 1 (rename), got: $apply" >&2
  exit 2
fi

shopt -s nullglob
sources=()
targets=()
for source in "$run_dir"/global_step_*; do
  [[ -d "$source" ]] || continue
  name=${source##*/}
  [[ "$name" =~ ^global_step_([0-9]+)$ ]] || continue
  step=${BASH_REMATCH[1]}

  # Keep global_step_0, global_step_100, global_step_200, and so on.
  (( 10#$step % 100 == 0 )) && continue

  target="${source}_skip"
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing target: $target" >&2
    exit 1
  fi
  sources+=("$source")
  targets+=("$target")
done

if (( ${#sources[@]} == 0 )); then
  echo "No non-hundred global_step_<N> directories found under: $run_dir"
  exit 0
fi

for index in "${!sources[@]}"; do
  printf '%s: %s -> %s\n' \
    "$( [[ "$apply" == 1 ]] && echo rename || echo preview )" \
    "${sources[index]##*/}" "${targets[index]##*/}"
done

if [[ "$apply" != 1 ]]; then
  echo "Preview only. Re-run with APPLY=1 to rename ${#sources[@]} checkpoint(s)."
  exit 0
fi

for index in "${!sources[@]}"; do
  mv -- "${sources[index]}" "${targets[index]}"
done
echo "Renamed ${#sources[@]} checkpoint(s); only global_step_<multiple of 100> remain discoverable."
