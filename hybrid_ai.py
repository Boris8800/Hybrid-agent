#!/usr/bin/env python3
"""
hybrid_ai.py — minimal OpenAI-compatible chat caller used by hybrid.sh.
Supports ANY base URL/model/key, so the ONLINE side can be DeepSeek, Groq, etc.
Local side is driven by Hermes Agent (its own configured default model).

Usage:
  hybrid_ai.py <base_url> <api_key> <model> <system_file> <user_file>
Reads the two prompt files, calls the chat completions endpoint, prints the reply.
"""
import json
import sys
import urllib.request

def main():
    if len(sys.argv) != 6:
        sys.stderr.write("usage: hybrid_ai.py <base_url> <api_key> <model> <system_file> <user_file>\n")
        return 1
    base, key, model, sysf, userf = sys.argv[1:6]
    with open(sysf) as f:
        system = f.read()
    with open(userf) as f:
        user = f.read()
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read().decode("utf-8"))
        print(d["choices"][0]["message"]["content"])
        return 0
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"online AI error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
