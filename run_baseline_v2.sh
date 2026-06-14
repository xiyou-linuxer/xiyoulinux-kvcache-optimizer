#!/bin/bash
# run_baseline_v2.sh — baseline v2 + FlexKV 一键启动
#
# 用法:
#   bash run_baseline_v2.sh                      # 完整 baseline
#   bash run_baseline_v2.sh --modes miniflex      # 只跑 FlexKV
#   bash run_baseline_v2.sh --modes baseline,miniflex  # 都跑
#   bash run_baseline_v2.sh --dry-run              # 预览命令
#   bash run_baseline_v2.sh --resume               # 断点续跑
#   bash run_baseline_v2.sh --repeat-count 1       # 快速版（1 次）
#   bash run_baseline_v2.sh --skip-sanity           # 跳过 sanity check
#   bash run_baseline_v2.sh --scenarios short_synth # 只跑指定场景
#
# 环境变量:
#   MODEL_PATH=/path/to/model bash run_baseline_v2.sh
#   LONGBENCH_DIR=/path/to/longbench bash run_baseline_v2.sh
#   REPEAT_COUNT=1 bash run_baseline_v2.sh
#   WEBHOOK_URL=https://... bash run_baseline_v2.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================"
echo "  baseline v2 + FlexKV 自动化测试"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  模型: ${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen3-8B}"
echo "  LongBench: ${LONGBENCH_DIR:-/root/autodl-tmp/longbench_local}"
echo "============================================"

ARGS=()

[ -n "${MODEL_PATH:-}" ] && ARGS+=(--model-path "$MODEL_PATH")
[ -n "${LONGBENCH_DIR:-}" ] && ARGS+=(--longbench-dir "$LONGBENCH_DIR")
[ -n "${REPEAT_COUNT:-}" ] && ARGS+=(--repeat-count "$REPEAT_COUNT")
[ -n "${WEBHOOK_URL:-}" ] && ARGS+=(--notify-webhook "$WEBHOOK_URL")
[ -n "${VLLM_EXTRA_ARGS:-}" ] && ARGS+=(--vllm-extra-args "$VLLM_EXTRA_ARGS")

ARGS+=("$@")

python3 tests/run_baseline_v2.py "${ARGS[@]}"
rc=$?

echo ""
echo "============================================"
echo "  结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  退出码: $rc"
echo "============================================"
exit $rc
