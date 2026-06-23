#!/usr/bin/env bash
set -euo pipefail

run_smoke=0
run_qwen3=0
run_qwen35=0

usage() {
  cat <<'EOF'
Usage: bash runs/check.sh [--smoke] [--qwen3] [--qwen35] [--qwen]

Default:
  compile all Python files and run the pytest suite.

Options:
  --smoke   Also run local train/eval/SFT/DPO/HF export-import and a temp web health check.
  --qwen3   Also validate local models/Qwen/Qwen3-0.6B dry-run, short generation, and parity.
  --qwen35  Also validate local models/Qwen/Qwen3.5-0.8B dry-run and short text generation.
  --qwen    Validate both downloaded Qwen3-0.6B and Qwen3.5-0.8B models.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      run_smoke=1
      ;;
    --qwen3)
      run_qwen3=1
      ;;
    --qwen35)
      run_qwen35=1
      ;;
    --qwen)
      run_qwen3=1
      run_qwen35=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

echo "[check] compile"
uv run python -m compileall nanoqwen scripts tests

echo "[check] pytest"
uv run pytest -q

if [[ "$run_smoke" == "1" ]]; then
  echo "[check] base train smoke"
  bash runs/smoke.sh

  echo "[check] eval smoke"
  bash runs/eval_smoke.sh

  echo "[check] report smoke"
  bash runs/report_smoke.sh

  echo "[check] sft smoke"
  bash runs/sft_smoke.sh

  echo "[check] dpo smoke"
  bash runs/dpo_smoke.sh

  echo "[check] hf local smoke"
  bash runs/hf_local_smoke.sh

  echo "[check] temporary web health"
  mkdir -p out
  uv run python scripts/chat_web.py \
    --checkpoint out/smoke \
    --host 127.0.0.1 \
    --port 8011 \
    --device cpu > out/check-chat-web.log 2>&1 &
  web_pid=$!
  cleanup() {
    kill "$web_pid" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  sleep 2
  curl -fsS http://127.0.0.1:8011/healthz
  echo
  cleanup
  trap - EXIT
fi

if [[ "$run_qwen3" == "1" ]]; then
  if [[ ! -f models/Qwen/Qwen3-0.6B/model.safetensors ]]; then
    echo "Qwen3-0.6B is missing. Run: bash runs/download_qwen3_06b.sh" >&2
    exit 1
  fi

  echo "[check] qwen3 dry smoke"
  bash runs/qwen3_dry_smoke.sh

  echo "[check] qwen3 text smoke"
  bash runs/qwen3_text_smoke.sh

  echo "[check] qwen3 compare smoke"
  bash runs/qwen3_compare_smoke.sh
fi

if [[ "$run_qwen35" == "1" ]]; then
  if [[ ! -f models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors ]]; then
    echo "Qwen3.5-0.8B is missing. Run: bash runs/download_qwen35_08b.sh" >&2
    exit 1
  fi

  echo "[check] qwen35 dry smoke"
  bash runs/qwen35_dry_smoke.sh

  echo "[check] qwen35 text smoke"
  bash runs/qwen35_text_smoke.sh

  echo "[check] qwen35 compare smoke"
  bash runs/qwen35_compare_smoke.sh
fi

echo "[check] ok"
