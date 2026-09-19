#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_train.sh configs/grpo/example.yaml [Hydra overrides...]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
config_root=$(realpath "$project_root/configs")
test_config_root=$(realpath "$project_root/tests/configs")
config_argument=$1
shift

if [[ "$config_argument" == /* ]]; then
  config_candidate=$config_argument
else
  config_candidate=$project_root/${config_argument#./}
fi
if [[ ! -f "$config_candidate" ]]; then
  echo "Training config does not exist: $config_candidate" >&2
  exit 2
fi
config_file=$(realpath "$config_candidate")
case "$config_file" in
  "$config_root"/*)
    selected_config_root=$config_root
    hydra_searchpath="file://$config_root,pkg://verl.trainer.config"
    ;;
  "$test_config_root"/*)
    selected_config_root=$test_config_root
    hydra_searchpath="file://$test_config_root,file://$config_root,pkg://verl.trainer.config"
    ;;
  *)
    echo "Training configs must be under $config_root or $test_config_root" >&2
    exit 2
    ;;
esac

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available; activate the server-aligned verl environment" >&2
  exit 1
fi

vendored_verl_root=$project_root/verl
verl_root=${R2OPL_VERL_ROOT:-$vendored_verl_root}
if [[ ! -f "$verl_root/verl/__init__.py" ]]; then
  echo "VERL source tree does not exist at: $verl_root" >&2
  exit 1
fi

export PYTHONPATH="$project_root:$verl_root${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

config_name=${config_file#"$selected_config_root"/}
config_name=${config_name%.*}
cd "$project_root"

exec python -m utils.training_entrypoint \
  --config-path "$selected_config_root" \
  --config-name "$config_name" \
  "hydra.searchpath=[$hydra_searchpath]" \
  "$@"
