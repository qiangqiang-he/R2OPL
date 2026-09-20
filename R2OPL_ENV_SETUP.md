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

## 没有 conda 时的三种安装方式

本环境只依赖 Python 3.12 + pip，conda 不是必需品。

### 方式一：uv（最推荐：无需 root、无需预装 Python，速度最快）

```bash
# 1) 安装 uv（单文件，不需要 root）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) 虚拟环境（uv 自动下载独立的 CPython 3.12）
cd /path/to/R2OPL
uv venv ~/.venvs/r2opl --python 3.12
source ~/.venvs/r2opl/bin/activate

# 3) 依赖（与手动安装第 2~4 步一致，pip 换成 uv pip）
uv pip install -i https://pypi.org/simple vllm==0.29.0
uv pip install -i https://pypi.org/simple "ray[default]==2.58.0"
uv pip install -i https://pypi.org/simple -e ./verl
uv pip install -i https://pypi.org/simple --no-deps torchvision
uv pip install -i https://pypi.org/simple --no-deps TransferQueue==0.1.7
uv pip install -i https://pypi.org/simple math-verify pytest py-spy

# 4) 仅 WSL2：写入 venv 的 activate（等效 conda 的 activate.d）
echo 'export VLLM_USE_V2_MODEL_RUNNER=0' >> ~/.venvs/r2opl/bin/activate
```

第 3 步等价写法：`uv pip install -r requirements.txt` 后补 `-e ./verl` 与两条 `--no-deps`。

### 方式二：先装 Miniconda（两条命令），现有脚本原样可用

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
# 之后： bash scripts/setup_r2opl_env.sh
```

### 方式三：机器已有 Python 3.12，直接 venv

```bash
python3.12 -m venv ~/.venvs/r2opl
source ~/.venvs/r2opl/bin/activate
pip install -U pip
# 之后执行"手动安装"的第 2~6 步（步骤编号不变）
```

> 注意：Ubuntu 22.04 系统自带 Python 3.10，不满足 verl/tensordict 的版本要求；
> 机器上没有 3.12 时请用方式一或方式二。

## 驱动要求与旧驱动服务器（如 535 / CUDA 12.2）

两套环境对应两种驱动代际，**先看 `nvidia-smi` 右上角 CUDA Version 再选**：

| 环境 | 组合 | 驱动要求 | 模型家族 | 验证状态 |
| --- | --- | --- | --- | --- |
| `r2opl`（默认） | vllm 0.29.0 + torch 2.13.0+cu130 | **≥ 580（CUDA 13）** | Qwen3 / Qwen3.5 / Qwen3.6 / **gemma-4** 全支持 | 本地 4 家族 + 6 算法 smoke 通过 |
| `r2opl-cu12`（旧驱动变体） | **vllm 0.19.1 + torch 2.10.0+cu128 + transformers 5.9.0** | ≥ 525（CUDA 12.x，含 535） | Qwen3 ✅、**gemma-4 ✅**；Qwen3.5 ✗、Qwen3.6 未测 | 本地 4090 实测通过（见下） |
| `r2opl-sgl`（旧驱动 SGLang 备选） | **sglang 0.5.11 + sglang-kernel 0.4.1 + torch 2.9.1+cu128** | ≥ 525（CUDA 12.x，含 535） | **gemma-4 ✅**（SGLang 引擎备选路径） | 本地 4090 实测通过（见下） |

**实测边界（2026-09-20，逐版验证 `_C` 扩展链接的 CUDA 运行库）**：
- vllm **0.19.1 是最后一个 cu12 构建**；0.20.0 起 `vllm._C` 链接 `libcudart.so.13`（0.20.2/0.21.0 实测 ImportError），0.22.0+ 依赖直接声明 cu13 内核包。
- **gemma-4 在 cu12 栈上可用（已实测）**：vllm 0.19.1 原生认识 `Gemma4ForConditionalGeneration`
  （早于官方博客宣称的 0.22），本地 2题×4rollout 单次 generate 通过，~248 tok/s
  （TRITON_ATTN，0.42 显存上限下）；测试脚本 `tests/verify_gemma4_vllm019_fallback.py`，
  报告 `tests/artifacts/vllm_family_checks/gemma4_vllm021_fallback.json`。
- Qwen3-0.6B 同栈实测 ~455 tok/s（FLASH_ATTN），脚本 `tests/verify_qwen3_vllm019.py`。
- Qwen3.5-2B 在 0.19.1 失败：vllm 的 speculator-config 路径把本地路径当 HF repo id
  （`HFValidationError`），属旧栈兼容 bug，非架构缺失；Qwen3.6 未测。

**要全家族（Qwen3.5/3.6）+ 满血性能，仍推荐**：
1. 升级驱动到 ≥ 580（H100 装 NVIDIA datacenter 驱动，一次性，最干净）——之后 `r2opl` 环境原样可用；
2. H100（数据中心卡）上尝试 NVIDIA **cuda-compat 前向兼容包**（`cuda-compat-13-x`）：
   ```bash
   # NVIDIA repo 安装后，让旧驱动加载 CUDA 13 用户态驱动
   export LD_LIBRARY_PATH=/usr/local/cuda-13/compat:$LD_LIBRARY_PATH
   ```
   官方仅对数据中心卡提供此机制，不保证成功，值得先于升驱动花 10 分钟试。

`r2opl-cu12` 变体安装（535 驱动服务器；注意 0.19.1 与 torch 2.10 配对）：

```bash
conda create -n r2opl-cu12 python=3.12 -y && conda activate r2opl-cu12
pip install -i https://pypi.org/simple "torch==2.10.0"          # PyPI 默认即 cu128 构建
pip install -i https://pypi.org/simple "vllm==0.19.1"           # 勿装 0.20+（cu13 重编译）
pip install -i https://pypi.org/simple "transformers==5.9.0"
pip install -i https://pypi.org/simple -e ./verl
pip install -i https://pypi.org/simple --no-deps TransferQueue==0.1.7
```

## `r2opl-sgl`：旧驱动上 Gemma 4 的 SGLang 备选路径（已实测通过）

若想要 SGLang 引擎（verl 同样支持 sglang rollout），gemma-4 在 cu12 上也可行：
sglang 0.5.11（2026-05-05，Gemma 4 首个支持版）的 Python 层不绑定 CUDA 大版本，CUDA 内核
在独立包 sglang-kernel 中——**0.4.1 是最后一个 cu12 构建**（0.4.2+ 链接
`libnvrtc.so.13`，实测 ImportError）。本地实测 gemma-4-E2B：2题×4rollout 单次
generate，1934 tokens / 2.49s（~776 tok/s，FLASH_ATTN 后端）；验证脚本
`tests/verify_sglang_gemma4.py`，报告 `tests/artifacts/vllm_family_checks/gemma4_sglang_cu12.json`。

```bash
conda create -n r2opl-sgl python=3.12 -y && conda activate r2opl-sgl
# 1) 先装 0.5.10 拉入全套 cu12 依赖（torch 2.9.1+cu128、sglang-kernel 0.4.1）
pip install -i https://pypi.org/simple sglang==0.5.10
# 2) Python 层升到带 Gemma 4 的 0.5.11（--no-deps 保住 cu12 依赖集）
pip install -i https://pypi.org/simple --no-deps sglang==0.5.11
# 3) Gemma 4 config 需要 transformers 5.6.0
pip install -i https://pypi.org/simple transformers==5.6.0
# 4) 把 sglang-kernel>=0.4.2 硬断言降级为警告（实测 0.4.1 对 Gemma 4 不缺算子）
python - <<'EOF'
import pathlib
import sglang.srt.utils.common as common
path = pathlib.Path(common.__file__)
src = path.read_text(encoding="utf-8")
if "[r2opl-cu12-patch]" in src:
    print("already patched")
else:
    old = """        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            raise Exception("""
    new = """        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            print(f"[r2opl-cu12-patch] {pkg} {installed_version} < {min_version}; continuing with cu12 kernel fallback")
            return
            raise Exception("""
    assert old in src
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")
EOF
# 自检：python tests/verify_sglang_gemma4.py
```

SGLang 注意事项：SamplingParams 无 vLLM 式 `n`——一个 question 的 n 条 rollout 通过
**重复 prompt n 次拼进同一次 `generate`** 实现（verl 的 sglang rollout 同款）；入口脚本
必须带 `if __name__ == "__main__":` 保护（spawn 子进程 re-import 主文件）。

## 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `r2opl_base` gemma 训练阶段 OOM（单块 ~4.4GB logits 分配失败但显存名义有空） | CUDA 分配器碎片化 → 运行前 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（已实测修复，全矩阵通过） |
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
