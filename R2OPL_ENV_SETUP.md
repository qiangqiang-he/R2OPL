# r2opl 环境安装指南

一条命令重建 R²OPL 的 `r2opl` 环境（vLLM 栈 + 本仓库 verl）：

```bash
cd /path/to/R2OPL
bash scripts/setup_r2opl_env.sh
```

脚本幂等：环境已存在则复用，pip 安装可重复执行。装完 `conda activate r2opl` 即可。

## 产出版本（2026-09-20 实测组合）

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| Python | 3.12.13 | conda |
| torch | 2.13.0+cu130 | 由 vllm==0.29.0 自动带入（含 CUDA 13 运行库 wheel，无需装 CUDA Toolkit） |
| **vllm** | **0.29.0** | 官方 PyPI |
| transformers | 5.10.4 | pip 自动收敛：vllm 要求 ≥5.10.4，verl 要求 <5.11 |
| ray | 2.58.0 | 官方 PyPI |
| torchvision | 0.28.0 | `--no-deps`，与 torch 2.13 配套 |
| TransferQueue | 0.1.7 | `--no-deps`（其 metadata 要求 numpy<2，与本环境 numpy 2.x 声明冲突但实测共存正常） |
| verl | 0.10.0.dev0 | editable → `<repo>/verl`（已合并上游 main 的 vendored 副本） |

精确到每个传递依赖的快照见 `verl/verl-0.10.0.dev0-r2opl-merged-vllm0.29.0.freeze.txt`（注意其中含本机 editable 路径行，不能直接 `pip install -r`，仅作审计用）。

## 前置条件

1. Linux 或 WSL2（Ubuntu 22.04 实测），NVIDIA 驱动 ≥ CUDA 13 UMD（`nvidia-smi` 右上角 CUDA Version ≥ 13.0；WSL 下驱动装在 Windows 侧）。
2. conda（miniconda/anaconda 均可）。
3. 网络：能访问 `pypi.org`。国内直连慢时用代理（见下）。

## 代理（国内网络推荐）

阿里云等镜像对该栈的大 wheel 只有 ~300KB/s；代理下官方 PyPI 实测 ~40MB/s（vllm 主 wheel 316MB / 8 秒）。

```bash
# WSL2 NAT 模式：代理跑在 Windows 上且开启"允许局域网连接"，
# 地址 = Windows 宿主网关 IP（ip route show default 第三列）+ 端口
USE_PROXY=1 PROXY_ADDR=http://172.27.112.1:7897 bash scripts/setup_r2opl_env.sh
```

## WSL2 专属行为（脚本自动处理）

- 检测到 WSL（`/proc/version` 含 microsoft）时，写入
  `etc/conda/activate.d/vllm_wsl.sh`：`export VLLM_USE_V2_MODEL_RUNNER=0`。
  原因：vLLM 0.29 默认的 V2 model runner 依赖 CUDA UVA（cuMem* 驱动 API），
  WSL 不支持，会直接 `RuntimeError: UVA is not available`；回退 V1 runner 即可。
  **裸机服务器（8 卡 H100）不会写也不需要此开关**，保持 V2 默认。
- 直接用绝对路径调用该环境的 python（不经 `conda activate`）时，务必
  `export PATH=$(conda info --base)/envs/r2opl/bin:$PATH`，否则 FlashInfer JIT
  找不到 `ninja`（EngineCore 启动报 `FileNotFoundError: 'ninja'`）。

## 安装后自检

```bash
conda activate r2opl
cd /path/to/R2OPL

# 1) CPU 级导入检查（脚本尾部已自动跑过一遍）
python -c "from verl.trainer import main_ppo_sync; print('OK')"

# 2) GPU rollout 自检：Qwen3 / Qwen3.5 / Qwen3.6 / gemma-4 各挑最小模型，
#    每家族 2 题 × 4 rollouts 单次 generate（模型路径在脚本内，可按机器改）
python tests/verify_vllm_model_families.py --family qwen3
python tests/verify_vllm_model_families.py --family qwen35
python tests/verify_vllm_model_families.py --family qwen36
python tests/verify_vllm_model_families.py --family gemma4

# 3) 算法 smoke（示例：GRPO 在线循环）
python tests/run_grpo_single_gpu_smoke.py \
  --config tests/configs/grpo/grpo_qwen3_1p7b_single_gpu_smoke.yaml
```

## 手动安装（与脚本完全等价的逐条指令）

适合逐步执行或排错时单步重放。假设：conda 已装、NVIDIA 驱动就绪、仓库位于 `/path/to/R2OPL`。

```bash
# 0)（可选，国内网络）代理 + 官方 PyPI；WSL2 下宿主 IP = ip route show default 第三列
export http_proxy=http://<宿主IP>:7897
export https_proxy=http://<宿主IP>:7897
export no_proxy=localhost,127.0.0.1
PIP_INDEX="https://pypi.org/simple"

# 1) 创建环境
conda create -n r2opl python=3.12 -y
conda activate r2opl          # 之后任何 vLLM 程序都要求 PATH 含本环境 bin（ninja 所在）

# 2) vLLM 栈（torch 2.13.0+cu130 与 CUDA 13 运行库 wheel 自动带入）
pip install -i $PIP_INDEX vllm==0.29.0
pip install -i $PIP_INDEX "ray[default]==2.58.0"

# 3) verl（editable，指向本仓库 vendored 副本；transformers 自动落在 5.10.4）
pip install -i $PIP_INDEX -e ./verl

# 4) 配套（注意两条 --no-deps）
pip install -i $PIP_INDEX --no-deps torchvision            # 与 torch 2.13 配套
pip install -i $PIP_INDEX --no-deps TransferQueue==0.1.7   # 避开其 numpy<2 假冲突
pip install -i $PIP_INDEX math-verify pytest py-spy

# 5) 仅 WSL2 需要（裸机服务器跳过！）
mkdir -p $(conda info --base)/envs/r2opl/etc/conda/activate.d
cat > $(conda info --base)/envs/r2opl/etc/conda/activate.d/vllm_wsl.sh <<'EOF'
export VLLM_USE_V2_MODEL_RUNNER=0
EOF

# 6) 验证
python -c "import torch, vllm, transformers, ray; print(torch.__version__, vllm.__version__, transformers.__version__, ray.__version__)"
python -c "from verl.trainer import main_ppo_sync; print('verl OK')"
python -c "import transfer_queue; print('TransferQueue OK')"
```

等价的 requirements 方式（步骤 2+4 可替换为）：

```bash
pip install -i $PIP_INDEX -r requirements.txt        # 顶层依赖（推荐）
# 或精确复现每个包版本：
pip install -i $PIP_INDEX -r requirements-freeze.txt # 完整快照（246 个 pin）
# 两种方式之后，都要补第 3 步的 verl 和第 4 步的 TransferQueue（--no-deps）。
```

## 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `RuntimeError: UVA is not available` | WSL 上用了 V2 runner → 确认 activate.d 里的 `VLLM_USE_V2_MODEL_RUNNER=0` 生效（或临时 export） |
| `FileNotFoundError: 'ninja'` | PATH 缺环境 bin 目录（FlashInfer JIT）→ 见上节 PATH 说明 |
| pip 解析出 transformers ≠ 5.10.x | 手动钉过其它版本 → `pip install -i https://pypi.org/simple "transformers>=5.10.4,<5.11"` |
| `pip check` 报 TransferQueue numpy 上界 | 预期现象（`--no-deps` 安装），可忽略；运行时已验证正常 |
| Qwen3.6-27B-FP8 启动报 Mamba cache blocks 不足 | 48GB 卡上需 `max_num_seqs ≤ ~169`（混合 Mamba 每 decode 序列占一块） |
| 下载极慢 | 走代理 + 官方 PyPI（镜像大 wheel 限速） |

## 服务器（8 卡）部署差异

1. 不需要 `VLLM_USE_V2_MODEL_RUNNER=0`（脚本不会写）。
2. verl 指向服务器仓库里的 vendored 副本（同一个 merge 结果），`pip install -e <repo>/verl` 即可。
3. 其余版本组合与本地一致；分布式（多卡）行为未在本地单卡验证过，首次上线建议先跑一个单卡 smoke 再上 8 卡正式配置。
