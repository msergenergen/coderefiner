#!/usr/bin/env bash
# OpenRouter run orchestrator (main experiment: docs/ablation.md §7).
#
#   scripts/run_openrouter.sh smoke              # pipeline on 2 problems (foreground)
#   scripts/run_openrouter.sh pilot [N]          # baseline, first N problems × 5 samples (default 20)
#   scripts/run_openrouter.sh baseline [N]       # baseline, N problems × 5 samples (default all)
#   scripts/run_openrouter.sh run [N] [SHARDS]   # pipeline, N problems × {verdict, rubric},
#                                                # SHARDS processes per design (default all, 4)
#   scripts/run_openrouter.sh status             # progress + spend
#   scripts/run_openrouter.sh langfuse-check     # verify the optional Langfuse keys
#   scripts/run_openrouter.sh judge-replay MODEL PROVIDER NAME --run-ids ...
#                                                # re-judge recorded public-passing attempts
#                                                # with another judge (benchmark/judge_replay.py)
#
# Settings: config/openrouter.env (tracked). Keys: only OPENROUTER_API_KEY and the
# optional LANGFUSE_* keys are read from .env.openrouter (git-ignored). The .env symlink is not touched.
# Extra env vars (SKIP_QIDS, DESIGNS="verdict", RESUME=file, LOG_TAG=_x for
# distinct log names, FEEDBACK_INCLUDE_CODE=false for feedback-only repair) pass through.
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source config/openrouter.env; set +a
KEY_FILE=${KEY_FILE:-.env.openrouter}
[ -f "$KEY_FILE" ] || { echo "Missing $KEY_FILE (copy .env.openrouter.example)"; exit 1; }
OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' "$KEY_FILE" | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')
[ -n "$OPENROUTER_API_KEY" ] || { echo "OPENROUTER_API_KEY is empty in $KEY_FILE"; exit 1; }
export OPENROUTER_API_KEY
# Optional Langfuse keys (models/tracing.py); tracing stays off when empty.
for VAR in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_BASE_URL; do
  VAL=$(grep -E "^${VAR}=" "$KEY_FILE" | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r' || true)
  [ -n "$VAL" ] && export "$VAR=$VAL"
done
echo "Langfuse tracing: $([ -n "${LANGFUSE_PUBLIC_KEY:-}" ] && [ -n "${LANGFUSE_SECRET_KEY:-}" ] && echo on || echo off)"

PY=${PY:-.venv/bin/python}
LOGDIR=outputs/openrouter/logs
mkdir -p "$LOGDIR"

spend() {
  "$PY" - <<'PY'
from models.budget import spent_usd
from config import settings
print(f"spent ${spent_usd():.4f} of ${settings.api_budget_usd:.2f}")
PY
}

case "${1:-}" in
  smoke)
    MAX_PROBLEMS=2 JUDGE_DECISION=verdict "$PY" main.py 2>&1 | tee "$LOGDIR/smoke.log"
    spend
    ;;
  pilot|baseline)
    N=${2:-$([ "$1" = pilot ] && echo 20 || echo 0)}
    MAX_PROBLEMS=$N "$PY" -m benchmark.sample_baseline --n 5 ${RESUME:+--resume "$RESUME"} \
      2>&1 | tee "$LOGDIR/${1}_n${N}.log"
    spend
    ;;
  run)
    N=${2:-0}; SHARDS=${3:-4}
    for DESIGN in ${DESIGNS:-verdict rubric}; do
      for i in $(seq 0 $((SHARDS-1))); do
        MAX_PROBLEMS=$N JUDGE_DECISION=$DESIGN SHARD_COUNT=$SHARDS SHARD_INDEX=$i \
          nohup "$PY" main.py > "$LOGDIR/${DESIGN}${LOG_TAG:-}_n${N}_shard$i.log" 2>&1 &
        sleep 2   # distinct second-stamped run_id per process
      done
    done
    echo "Started. Logs: $LOGDIR   Check: $0 status"
    ;;
  judge-replay)
    MODEL=$2; PROVIDER=$3; NAME=$4; shift 4
    OPENROUTER_JUDGE_MODEL=$MODEL OPENROUTER_JUDGE_PROVIDER=$PROVIDER \
      "$PY" -m benchmark.judge_replay --name "$NAME" "$@" 2>&1 | tee "$LOGDIR/judge_replay_$NAME.log"
    spend
    ;;
  langfuse-check)
    "$PY" - <<'PY'
from models import tracing
ok = tracing.enabled() and tracing._client.auth_check()
print("Langfuse auth:", "OK" if ok else "FAILED or no keys (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
PY
    ;;
  status)
    for f in "$LOGDIR"/*.log; do
      [ -e "$f" ] && printf "%-40s %3s done %s\n" "$(basename "$f")" \
        "$(grep -c 'Problem done\|samples done' "$f" || true)" \
        "$(grep -q 'BudgetExceeded\|Traceback' "$f" && echo '<- ERROR, see log' || true)"
    done
    echo "running processes: $(pgrep -f 'main.py|sample_baseline' | wc -l | tr -d ' ')"
    spend
    ;;
  *) sed -n 2,18p "$0"; exit 1 ;;
esac
