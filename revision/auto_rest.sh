#!/bin/bash
# Continue after the user's approval of the PARTIAL small-test result: remaining training -> evaluation/analysis.
cd "$(dirname "$0")/.." || exit 1
LOG=revision_runs/auto_pipeline.log; echo "[auto] user approved continuation $(date)" >> $LOG
rm -f revision_runs/PAUSE
.venv/bin/python -u revision/run_queue.py --plan rest --parallel 2 >> revision_runs/queue.log 2>&1
[ -f revision_runs/PAUSE ] && { echo "[auto] queue paused $(date) -> stopped" >> $LOG; exit 0; }
for s in final_eval.py robustness.py interpretability.py make_results.py; do
  echo "[auto] running $s $(date)" >> $LOG; .venv/bin/python -u revision/$s >> $LOG 2>&1 || echo "[auto] $s FAILED (rc=$?)" >> $LOG
done
echo "[auto] ALL DONE $(date)" >> $LOG
