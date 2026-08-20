#!/bin/bash
# Reload the local LM Studio model with parallel=1 (single-sequence).
#
# WHY: LM Studio's batched prediction kit (parallel=4, the default) crashes on
# this MLX model with "ValueError: Slice indices must be 32-bit integers" when
# serving the OpenAI-compatible endpoint Kilo uses (/v1/chat/completions).
# Loading with --parallel 1 avoids the batched scheduler entirely and makes
# both the native /api/v1/chat and the OpenAI-compatible /v1/chat/completions
# endpoints work. See AGENTS.md → "Known pitfall: chat degeneration".

set -euo pipefail

MODEL="${LOCAL_MODEL:-qwen2.5-coder-14b-instruct-mlx}"
CTX="${LOCAL_CONTEXT:-16384}"

echo "[local] unloading ${MODEL} (if loaded)..."
lms unload "${MODEL}" >/dev/null 2>&1 || true

echo "[local] loading ${MODEL} with --parallel 1 -c ${CTX} ..."
lms load "${MODEL}" --parallel 1 -c "${CTX}" -y

echo "[local] verifying OpenAI-compatible chat endpoint..."
curl -s --max-time 60 http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"temperature\":0.2,\"max_tokens\":10}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('[local]', d.get('choices',[{}])[0].get('message',{}).get('content','') or d.get('error','OK'))"

echo "[local] done — model ready for Kilo chat."
