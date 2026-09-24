#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_ersr_eval.sh configs/ersr_eval/CONFIG.yaml [--validate-only] [--resume]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
config_root=$(realpath "$project_root/configs/ersr_eval")
test_config_root=
if [[ -d "$project_root/tests/configs/ersr_eval" ]]; then
  test_config_root=$(realpath "$project_root/tests/configs/ersr_eval")
fi

config_argument=$1
shift

if [[ "$config_argument" == /* ]]; then
  config_candidate=$config_argument
else
  config_candidate=$project_root/${config_argument#./}
fi
if [[ ! -f "$config_candidate" ]]; then
  echo "ERSR evaluation config does not exist: $config_candidate" >&2
  exit 2
fi
config_file=$(realpath "$config_candidate")
if [[ "$config_file" == "$config_root/"* ]]; then
  :
elif [[ -n "$test_config_root" && "$config_file" == "$test_config_root/"* ]]; then
  :
else
  echo "ERSR evaluation configs must be under $config_root${test_config_root:+ or $test_config_root}" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available; activate the server-aligned environment first" >&2
  exit 1
fi

vendored_verl_root=$project_root/verl
verl_root=${R2OPL_VERL_ROOT:-$vendored_verl_root}
if [[ ! -f "$verl_root/verl/__init__.py" ]]; then
  echo "VERL source tree does not exist at: $verl_root" >&2
  exit 1
fi

export PYTHONPATH="$project_root:$verl_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$project_root"
exec python -m utils.ersr_evaluation --config "$config_file" "$@"
