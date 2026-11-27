#!/usr/bin/env bash
# Silver HTR experiment, end to end: teacher labels for C, training B/C (pre-declared), validation scoring,
# page CER with the midterm detector, the optional H control and the exploratory M run (silver + human replay).
#
#   nohup setsid scripts/run_silver_htr.sh > results/htr/silver/log_driver.txt 2>&1 &
#   uv run python scripts/silver_report.py        # tables -> results/htr/silver/README.md
#
# Each stage logs to results/htr/silver/log_<stage>.txt and leaves done_<stage> when it succeeds, so rerunning the
# script skips finished stages. Watchdog: on the WSL2 laptop GPU jobs occasionally hang (GPU 0%, main process at
# 100% CPU) and never recover. A stage whose log is silent for 7 minutes is killed and retried, at most 8 times.
# H and M training resumes from its last evaluation (every 400 steps); other stages restart.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/htr/silver
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
    local name=$1; shift
    if [ -f "$OUT/done_$name" ]; then echo "$(date +%T) SKIP $name"; return 0; fi
    for attempt in 1 2 3 4 5 6 7 8; do
        echo "$(date +%T) START $name (attempt $attempt)"
        setsid .venv/bin/python -u scripts/silver_htr.py "$@" > "$OUT/log_$name.txt" 2>&1 &
        local pid=$!
        while kill -0 $pid 2>/dev/null; do
            sleep 30
            if [ $(( $(date +%s) - $(stat -c %Y "$OUT/log_$name.txt") )) -gt 420 ]; then
                echo "$(date +%T) STALLED $name, killing"; kill -- -$pid; sleep 10; kill -9 -- -$pid 2>/dev/null
                mv "$OUT/log_$name.txt" "$OUT/log_${name}_stalled_$(date +%H%M).txt"; break
            fi
        done
        if wait $pid && [ -f "$OUT/log_$name.txt" ]; then touch "$OUT/done_$name"; echo "$(date +%T) DONE $name"; return 0; fi
        sleep 15
    done
    echo "$(date +%T) FAILED $name"; return 1
}

stage teacher teacher || exit 1
stage train_B train B && stage eval_B evaluate B
stage train_C train C && stage eval_C evaluate C
stage eval_A evaluate A
stage pages pages A B C
stage train_H train H && stage eval_H evaluate H
stage train_M train M && stage eval_M evaluate M && stage pages_M pages A B C M
echo "$(date +%T) ALL DONE"
