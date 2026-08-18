#!/bin/bash
# check_embedding_service.sh - macOS/Linux health-check for an OpenAI-compatible
# embedding service (LM Studio / Ollama / local-ai). Read-only: it only probes
# GET /v1/models and POST /v1/embeddings. It does NOT restart or kill anything.
# Exit code: 0 = healthy, 1 = unhealthy.

BASE_URL="${EMBEDDING_URL:-http://127.0.0.1:1234}"
LOG_FILE="${EMBEDDING_LOG:-$HOME/embedding_service.log}"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE" >&2
}

# Extract model ids from /v1/models; tries jq, falls back to grep/sed.
get_models() {
    local body
    body="$(curl -s --max-time 5 "$BASE_URL/v1/models" 2>/dev/null)" || return 1
    [ -z "$body" ] && return 1
    local models=""
    if command -v jq >/dev/null 2>&1; then
        models="$(echo "$body" | jq -r '.data[].id? // .models[]? // .[]?' 2>/dev/null)"
    fi
    if [ -z "$models" ]; then
        # Fallback: pull "id":"..." fields without jq.
        models="$(echo "$body" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
            | sed 's/.*"[[:space:]]*:[[:space:]]*"//; s/"$//')"
    fi
    # Prefer embedding-looking models.
    local emb
    emb="$(echo "$models" | grep -iE 'embed|e5|bge|nomic|ada|text-embedding')"
    echo "${emb:-$models}"
}

test_embedding() {
    local model="$1"
    local code
    code="$(curl -s --max-time 10 -o /dev/null -w '%{http_code}' \
        -X POST "$BASE_URL/v1/embeddings" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"$model\", \"input\": \"test\"}" 2>/dev/null)"
    # 200 = works; 404 = service up but model unknown -> still usable.
    [ "$code" = "200" ] || [ "$code" = "404" ]
}

find_working_model() {
    local models
    models="$(get_models)"
    if [ -z "$models" ]; then
        log_message "no models detected; trying common names"
        models="text-embedding-nomic-embed-text-v1.5
text-embedding-ada-002
bge-small-en
e5-small-v2
all-MiniLM-L6-v2
intfloat/e5-small-v2"
    fi
    while IFS= read -r m; do
        [ -z "$m" ] && continue
        log_message "testing model: $m"
        if test_embedding "$m"; then
            echo "$m"
            return 0
        fi
    done <<< "$models"
    return 1
}

main() {
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    log_message "=== checking embedding service at $BASE_URL ==="

    local model
    model="$(find_working_model)"
    if [ -n "$model" ]; then
        log_message "service healthy; working model: $model"
        echo "Embedding service: healthy (model: $model)"
        exit 0
    fi

    # Service might be up but no working embedding model.
    if curl -s --max-time 3 -o /dev/null "$BASE_URL"; then
        log_message "service reachable but no working embedding model found"
        echo "Embedding service: reachable, but no working embedding model found"
        exit 0
    fi

    log_message "service not responding"
    echo "Embedding service: not responding"
    exit 1
}

main
