#!/usr/bin/env bash
# 自包含演示脚本(只此一个 .sh):内联启动 vLLM(MiniFlex/APC/叠加/SSD)+ 调用通用 bench_*.py。
#   bash demo.sh          # 每幕停顿(后期配字幕)
#   PAUSE=0 bash demo.sh  # 连续跑
# 机器相关项(路径/模型)可用环境变量覆盖:HF_HOME / MODEL / GPU_MEM / MINIFLEX_*
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ---- 环境(原 env.sh 折叠进来;按机器可覆盖)----
export PATH="/root/miniconda3/bin:$PATH"
export CPATH="${CPATH:-/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/include}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export MINIFLEX_MAX_MODEL_LEN="${MINIFLEX_MAX_MODEL_LEN:-32768}"
export MINIFLEX_NUM_CPU_BLOCKS="${MINIFLEX_NUM_CPU_BLOCKS:-8192}"
export ENABLE_MINIFLEX=1 PYTHONPATH=pysrc
export MINIFLEX_GPU_REGISTER_PORT="ipc:///tmp/miniflex_demo.sock"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GPU_MEM="${GPU_MEM:-0.80}"
KVCFG='{"kv_connector":"MiniFlexConnectorV1","kv_connector_module_path":"miniflex.integration.vllm.connector","kv_role":"kv_both"}'
URL="http://localhost:8000"; SLOG=/tmp/demo_serve.log; PAUSE="${PAUSE:-1}"
SSD_E2E_CPU_BLOCKS="${SSD_E2E_CPU_BLOCKS:-512}"
SSD_E2E_SSD_BLOCKS="${SSD_E2E_SSD_BLOCKS:-2048}"
SSD_E2E_BODY_REPEAT="${SSD_E2E_BODY_REPEAT:-250}"
SSD_E2E_NUM_PREFIXES="${SSD_E2E_NUM_PREFIXES:-4}"
SSD_E2E_ROUNDS="${SSD_E2E_ROUNDS:-3}"
SSD_E2E_CACHE_DIR="${SSD_E2E_CACHE_DIR:-${MINIFLEX_SSD_CACHE_DIR:-/root/autodl-tmp/miniflex-ssd-e2e}}"
SSD_E2E_USE_DIRECT_IO="${SSD_E2E_USE_DIRECT_IO:-${MINIFLEX_USE_DIRECT_IO:-1}}"
SSD_E2E_OUT="${SSD_E2E_OUT:-/tmp/miniflex_ssd_e2e.json}"
H='\033[1;36m';G='\033[1;32m';Y='\033[1;33m';R='\033[0m'
FRAME_INNER_WIDTH=66
banner(){
  local title="$*"
  local title_width padding border
  title_width=$(python3 -c 'import sys,unicodedata;print(sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in sys.argv[1]))' "$title")
  padding=$((FRAME_INNER_WIDTH - title_width - 1))
  (( padding < 1 )) && padding=1
  printf -v border '%*s' "$FRAME_INNER_WIDTH" ''
  border=${border// /═}
  printf '\n%b╔%s╗%b\n' "$H" "$border" "$R"
  printf '%b║ %s%*s║%b\n' "$H" "$title" "$padding" "" "$R"
  printf '%b╚%s╝%b\n' "$H" "$border" "$R"
}
step(){ echo -e "${Y}>>> $*${R}"; }
pause(){ [ "$PAUSE" = "1" ] && read -rs _ </dev/tty || true; }

show_ssd_summary(){
  python3 -c '
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
timings = data["timings"]
rounds = data["configuration_claim"]["rounds"]
cold = timings["cold"]["ttft_p50_ms"]
ssd = timings["ssd_hit"]["ttft_p50_ms"]
cpu = timings["cpu_hit_after_ssd"]["ttft_p50_ms"]
cold_over_ssd = timings["cold_over_ssd"]
cold_over_cpu = timings["cold_over_cpu"]
conclusion = "SSD 恢复快于重算" if timings["verdict"] == "ssd_faster_than_recompute" else "当前长度下重算更快，但 SSD 路径正确"
print(f"   [SSD E2E] 冷重算 p50={cold:.1f}ms | SSD 恢复 p50={ssd:.1f}ms | CPU 命中 p50={cpu:.1f}ms")
print(f"   [加速比] 冷/SSD={cold_over_ssd:.2f}x | 冷/CPU={cold_over_cpu:.2f}x")
print(f"   [结论] {conclusion}")
print(f"   ✅ {rounds}/{rounds} 轮完整 SSD 命中，并在回填后完整 CPU 命中")
' "$SSD_E2E_OUT"
}

# ---- 内联启动 vLLM。mode: miniflex | apc | both | ssd | baseline ----
serve(){
  pkill -9 -f "vllm serve" 2>/dev/null;pkill -9 -f "VLLM::EngineCore" 2>/dev/null;sleep 3;rm -f "$SLOG" /tmp/miniflex_demo.sock
  local a=(serve "$MODEL" --served-model-name qwen3-8b --disable-hybrid-kv-cache-manager
           --gpu-memory-utilization "$GPU_MEM" --max-model-len "$MINIFLEX_MAX_MODEL_LEN" --enforce-eager --port 8000)
  case "$1" in miniflex|both|ssd) a+=(--kv-transfer-config "$KVCFG");; esac
  case "$1" in apc|both) a+=(--enable-prefix-caching);; *) a+=(--no-enable-prefix-caching);; esac
  step "启动服务 [$1]"
  vllm "${a[@]}" > "$SLOG" 2>&1 & SPID=$!;disown 2>/dev/null||true
  for i in $(seq 1 150);do grep -q "Application startup complete" "$SLOG" 2>/dev/null&&{ sleep 18;step "服务就绪 [$1]";return 0;};kill -0 "$SPID" 2>/dev/null||{ echo "!!!退出";tail -20 "$SLOG";return 1;};sleep 3;done;echo "!!!超时";return 1; }

banner "MiniFlex × vLLM 0.23  ·  KV 多级缓存复用"
echo "   Qwen3-8B / RTX 5090 / vLLM 0.23"
pause

banner "① 功能验证"
serve miniflex || exit 1
step "补全请求:'The capital of France is'"
curl -s "$URL/v1/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3-8b","prompt":"The capital of France is","max_tokens":12,"temperature":0}' | python -c 'import sys,json;print("   ✅",repr(json.load(sys.stdin)["choices"][0]["text"]))'
pause

banner "② 长上下文加速:冷(重算) vs 热(MiniFlex 命中)"
PYTHONPATH=pysrc python bench_ttft.py --url "$URL" --model qwen3-8b --body-repeat 33 --runs 1 >/dev/null 2>&1
for br in 33 65 130 250 500 750 1000;do echo -e "${G}  ── 上下文 ~$(( (br*30+500)/1000 ))k ──${R}"
  PYTHONPATH=pysrc python bench_ttft.py --url "$URL" --model qwen3-8b --body-repeat "$br" --runs 3 2>&1|grep -E "冷\(|热\(|加速比";done
pause

banner "③ 超显存容量:工作集递增,MiniFlex vs vLLM 原生 APC(GPU 容量 ~68k)"
step "MiniFlex(CPU/SSD 多级):工作集 ~45k → ~75k → ~105k"
for n in 6 10 14;do
  PYTHONPATH=pysrc python bench_overflow.py --url "$URL" --tag MiniFlex --num-prefixes $n --rounds 2 > /tmp/mf_ovf_$n.txt 2>&1
  echo "   ~$((n*7500/1000))k:  $(grep -oP '^\[MiniFlex\].*' /tmp/mf_ovf_$n.txt)";done
PYTHONPATH=pysrc python bench_mixed.py --url "$URL" --tag miniflex > /tmp/mix_mf.txt 2>&1
pause
serve apc || exit 1
step "APC(vLLM 原生前缀缓存):同样工作集"
for n in 6 10 14;do
  PYTHONPATH=pysrc python bench_overflow.py --url "$URL" --tag APC --num-prefixes $n --rounds 2 > /tmp/apc_ovf_$n.txt 2>&1
  echo "   ~$((n*7500/1000))k:  $(grep -oP '^\[APC\].*' /tmp/apc_ovf_$n.txt)";done
PYTHONPATH=pysrc python bench_mixed.py --url "$URL" --tag apc > /tmp/mix_apc.txt 2>&1
pause
echo -e "${G}── 对比(median TTFT):越过 GPU 容量后 APC 崩、MiniFlex 稳 ──${R}"
printf "   %-9s %-12s %-12s\n" "工作集" "APC" "MiniFlex"
for n in 6 10 14;do printf "   ~%-8s %-12s %-12s\n" "$((n*7500/1000))k" "$(grep -oP 'median=\K[0-9]+ms' /tmp/apc_ovf_$n.txt)" "$(grep -oP 'median=\K[0-9]+ms' /tmp/mf_ovf_$n.txt)";done
pause

banner "④ APC + MiniFlex 叠加(混合负载:热数据 + 长尾)"
serve both || exit 1
PYTHONPATH=pysrc python bench_mixed.py --url "$URL" --tag both > /tmp/mix_both.txt 2>&1
step "三方对比(hot=热数据 tail=长尾 overall=总体,越低越好)"
echo -e "${G}"; grep -h '^\[' /tmp/mix_apc.txt /tmp/mix_mf.txt /tmp/mix_both.txt; echo -e "${R}"
pause

banner "⑤ 真实 SSD E2E: CPU 淘汰 → SSD 命中 → CPU 回填 → GPU 恢复"
mkdir -p "$SSD_E2E_CACHE_DIR" || exit 1
rm -f /tmp/miniflex_metrics.json "$SSD_E2E_OUT" /tmp/ssd_e2e.txt
export MINIFLEX_ENABLE_SSD=1
export MINIFLEX_NUM_CPU_BLOCKS="$SSD_E2E_CPU_BLOCKS"
export MINIFLEX_NUM_SSD_BLOCKS="$SSD_E2E_SSD_BLOCKS"
export MINIFLEX_SSD_CACHE_DIR="$SSD_E2E_CACHE_DIR"
export MINIFLEX_SSD_FILE_PREFIX="miniflex_demo_ssd"
export MINIFLEX_USE_DIRECT_IO="$SSD_E2E_USE_DIRECT_IO"
export MINIFLEX_EVICTION_POLICY=lru

serve ssd || exit 1
step "真实请求:${SSD_E2E_NUM_PREFIXES} 条独立前缀，CPU=${SSD_E2E_CPU_BLOCKS} blocks，SSD=${SSD_E2E_SSD_BLOCKS} blocks，O_DIRECT=${SSD_E2E_USE_DIRECT_IO}"
PYTHONPATH=pysrc python bench_ssd_e2e.py \
  --url "$URL" \
  --model qwen3-8b \
  --cpu-blocks "$SSD_E2E_CPU_BLOCKS" \
  --ssd-blocks "$SSD_E2E_SSD_BLOCKS" \
  --body-repeat "$SSD_E2E_BODY_REPEAT" \
  --num-prefixes "$SSD_E2E_NUM_PREFIXES" \
  --rounds "$SSD_E2E_ROUNDS" \
  --metrics-file /tmp/miniflex_metrics.json \
  --require-full-ssd-hit \
  --out "$SSD_E2E_OUT" > /tmp/ssd_e2e.txt 2>&1 || { echo "!!! SSD E2E 失败"; tail -40 /tmp/ssd_e2e.txt; tail -20 "$SLOG"; exit 1; }
step "结果汇总(TTFT p50，越低越好)"
echo -e "${G}"
show_ssd_summary
echo -e "${R}"
step "原始数据已保存: $SSD_E2E_OUT"
pause

pkill -9 -f "vllm serve" 2>/dev/null;wait 2>/dev/null
banner "完成"
