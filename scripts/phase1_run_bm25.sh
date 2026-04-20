#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_NAME=""
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "phase1_run_bm25.sh requires --run-name so it can invoke evaluation." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.retrieval.run_bm25 "${ARGS[@]}"
exec "$PYTHON_BIN" -m src.evaluation.evaluate --run-name "$RUN_NAME"
