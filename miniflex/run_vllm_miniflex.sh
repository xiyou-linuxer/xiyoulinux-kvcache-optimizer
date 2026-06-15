#!/usr/bin/env bash
# 启动 vLLM + MiniFlex KV connector(单机单卡)
# 用法:  bash run_vllm_miniflex.sh
# 环境变量:
#   MINIFLEX_DEBUG=1           — 开启调试日志
#   MINIFLEX_ENFORCE_EAGER=0   — 关闭 --enforce-eager（允许 CUDA graph）
#   MINIFLEX_MAX_MODEL_LEN     — 覆盖 max-model-len（默认 2048）
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen1.5-0.5B-Chat}"
PORT="${PORT:-8000}"
REGISTER_PORT="${MINIFLEX_GPU_REGISTER_PORT:-ipc:///tmp/miniflex_smoke.sock}"

cd "$(dirname "$(readlink -f "$0")")"

# ---- 1) 彻底清理上一次的残留(EngineCore / TransferManager 子进程 + stale socket)----
echo ">>> 清理残留进程和 socket ..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true   # vLLM 会改进程名，必须单独抓
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
sleep 2
rm -f /tmp/miniflex_*.sock
echo ">>> 当前显存占用:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---- 2) 启动(JSON 必须在同一行,不要折行)----
echo ">>> 启动 vLLM(connector=MiniFlexConnectorV1)..."
export ENABLE_MINIFLEX=1
export MINIFLEX_GPU_REGISTER_PORT="$REGISTER_PORT"
export PYTHONPATH=pysrc
export HF_HUB_OFFLINE=1          # 你的 socks:// 代理 vLLM 不支持,强制走本地缓存
export TRANSFORMERS_OFFLINE=1

# max-model-len 可通过环境变量覆盖（编排器会自动传入）
MAX_MODEL_LEN="${MINIFLEX_MAX_MODEL_LEN:-2048}"

# --enforce-eager: 默认开启（兼容 connector hook），可通过 MINIFLEX_ENFORCE_EAGER=0 关闭
EAGER_FLAG="--enforce-eager"
if [ "${MINIFLEX_ENFORCE_EAGER:-1}" != "1" ]; then
    EAGER_FLAG=""
    echo ">>> --enforce-eager 已关闭（CUDA graph 将启用）"
fi

exec vllm serve "$MODEL" \
  --kv-transfer-config '{"kv_connector":"MiniFlexConnectorV1","kv_connector_module_path":"miniflex.integration.vllm.connector","kv_role":"kv_both"}' \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.55 \
  --max-model-len "$MAX_MODEL_LEN" \
  $EAGER_FLAG \
  --port "$PORT"
