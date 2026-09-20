#!/usr/bin/env bash
# =============================================================================
# setup_r2opl_env.sh — 一键创建 R²OPL 的 `r2opl` conda 环境
#
# 产出环境（2026-09-20 实测组合）:
#   Python 3.12.13 / torch 2.13.0+cu130 / vllm 0.29.0 / transformers 5.10.4
#   ray 2.58.0 / tensordict 0.10.0 / TransferQueue 0.1.7 (--no-deps)
#   verl 0.10.0.dev0 (editable, 指向本仓库 vendored <repo>/verl)
#
# 用法（Linux / WSL2，需已安装 conda 与 NVIDIA 驱动）:
#   bash scripts/setup_r2opl_env.sh
#
# 可选环境变量:
#   ENV_NAME=r2opl            conda 环境名
#   USE_PROXY=1               走代理访问官方 PyPI（见 PROXY_ADDR）
#   PROXY_ADDR=http://172.27.112.1:7897
#                             代理地址；WSL2 NAT 模式下通常是 Windows 宿主
#                             网关 IP（ip route show default 的第三列），
#                             且代理软件需开启"允许局域网连接"。
#   SKIP_VERL=1               只装 vLLM 栈，不安装本仓库的 verl
#
# 说明:
#   * 强制使用官方 PyPI（-i https://pypi.org/simple）：国内镜像对该栈的
#     大 wheel（vllm 316MB / torch ~1GB / CUDA 库若干 GB）常常只有
#     ~300KB/s，代理下官方源可达 ~40MB/s。
#   * vllm==0.29.0 会自动带上配套的 torch==2.13.0+cu130 与 CUDA 13 运行库
#     wheel，无需单独安装 CUDA Toolkit。
#   * transformers 不要手动指定：vllm 0.29 要求 >=5.10.4，本仓库 verl 要求
#     >=5.5.3,<5.11，pip 会自动收敛到 5.10.4。
#   * TransferQueue 的 metadata 声明 numpy<2，与本环境 numpy 2.x 冲突，
#     因此沿用 --no-deps 安装（实测共存正常，算法 smoke 已验证）。
#   * WSL2 会自动写入 activate.d 开关（见脚本尾部注释）；裸机服务器
#     （如 8 卡 H100）不需要也不会写入该开关。
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-r2opl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
TRANSFERQUEUE_VERSION="${TRANSFERQUEUE_VERSION:-0.1.7}"
INDEX_URL="https://pypi.org/simple"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------- 0. 代理（可选） ----------
if [ "${USE_PROXY:-0}" = "1" ]; then
    PROXY_ADDR="${PROXY_ADDR:-http://172.27.112.1:7897}"
    export http_proxy="$PROXY_ADDR"
    export https_proxy="$PROXY_ADDR"
    export no_proxy="localhost,127.0.0.1"
    echo "[setup] using proxy: $PROXY_ADDR"
fi

# ---------- 1. 定位 conda ----------
if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$c/etc/profile.d/conda.sh"
            break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || {
    echo "[setup] ERROR: conda not found; install miniconda first." >&2
    exit 1
}

# ---------- 2. 创建/复用环境 ----------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] conda env '${ENV_NAME}' already exists, reusing it."
else
    echo "[setup] creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})..."
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi
conda activate "$ENV_NAME"
echo "[setup] python: $(python -V)"

# 后续所有 pip 调用的公共参数
pip_install() { python -m pip install -i "$INDEX_URL" "$@"; }

# ---------- 3. vLLM 栈（版本由此钉死，依赖由 pip 解析） ----------
echo "[setup] installing vllm==${VLLM_VERSION} (pulls torch/CUDA wheels)..."
pip_install "vllm==${VLLM_VERSION}"

echo "[setup] installing ray..."
pip_install "ray[default]>=2.41.0"

# ---------- 4. 本仓库 verl（editable） ----------
if [ "${SKIP_VERL:-0}" != "1" ]; then
    if [ -f "$REPO_ROOT/verl/setup.py" ]; then
        echo "[setup] installing vendored verl (editable) from $REPO_ROOT/verl ..."
        pip_install -e "$REPO_ROOT/verl"
    else
        echo "[setup] WARNING: $REPO_ROOT/verl/setup.py not found; skipping verl." >&2
    fi
fi

# ---------- 5. 配套小件 ----------
pip_install --no-deps "TransferQueue==${TRANSFERQUEUE_VERSION}"
pip_install --no-deps torchvision          # 与 vllm 带入的 torch 配套
pip_install math-verify pytest
pip_install py-spy                          # 排查卡顿用，可选但推荐

# ---------- 6. WSL2 专属开关 ----------
# vLLM >= 0.29 默认的 V2 model runner 依赖 CUDA UVA（cuMem* 驱动 API），
# WSL 不支持会直接 RuntimeError: UVA is not available；回退 V1 runner 即可。
# 裸机（服务器 8 卡）不要设置，保持默认 V2。
if grep -qi microsoft /proc/version 2>/dev/null; then
    ACT_DIR="$(conda info --base)/envs/${ENV_NAME}/etc/conda/activate.d"
    mkdir -p "$ACT_DIR"
    cat > "$ACT_DIR/vllm_wsl.sh" <<'EOF'
# vLLM 0.29's GPUModelRunnerV2 needs CUDA UVA (cuMem* driver APIs) which WSL
# does not support; fall back to the classic V1 model runner. Harmless if the
# vLLM version predates the V2 runner.
export VLLM_USE_V2_MODEL_RUNNER=0
EOF
    echo "[setup] WSL detected: wrote $ACT_DIR/vllm_wsl.sh (VLLM_USE_V2_MODEL_RUNNER=0)"
fi

# ---------- 7. 验证 ----------
echo "[setup] verifying imports..."
python - <<'PY'
import torch, vllm, transformers, ray
print("torch        ", torch.__version__)
print("vllm         ", vllm.__version__)
print("transformers ", transformers.__version__)
print("ray          ", ray.__version__)
assert vllm.__version__.startswith("0.29"), "unexpected vllm version"
assert transformers.__version__.startswith("5.10"), "unexpected transformers version"
PY
if [ "${SKIP_VERL:-0}" != "1" ] && [ -f "$REPO_ROOT/verl/setup.py" ]; then
    python -c "from verl.trainer import main_ppo_sync; print('verl         ', 'main_ppo_sync import OK')"
fi
python -c "import transfer_queue; print('TransferQueue', transfer_queue.__version__)"

echo
echo "[setup] DONE. 激活方式:  conda activate ${ENV_NAME}"
echo "[setup] GPU rollout 自检（需本地模型目录）:"
echo "[setup]   python tests/verify_vllm_model_families.py --family gemma4"
echo "[setup] 注意: 直接用绝对路径调用该环境 python 时，请先"
echo "[setup]       export PATH=\$(conda info --base)/envs/${ENV_NAME}/bin:\$PATH"
echo "[setup]       （FlashInfer JIT 需要 bin 目录里的 ninja）"
