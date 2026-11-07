#!/bin/bash
# Local auto-pipeline (no Claude involvement): small-test decision -> (GO only) remaining training -> evaluation/analysis.
cd "$(dirname "$0")/.." || exit 1
LOG=revision_runs/auto_pipeline.log; echo "[auto] start $(date)" >> $LOG
while [ ! -f revision_runs/models/baseline_seed7.zip ] || [ ! -f revision_runs/models/baseline_seed107.zip ] || pgrep -f revision/tune_classical.py >/dev/null; do sleep 60; done
echo "[auto] small test + tuning finished $(date)" >> $LOG
.venv/bin/python -u revision/smalltest.py >> $LOG 2>&1
DEC=$(python3 -c "import json;d=json.load(open('revision_runs/analysis/smalltest_decision.json'));print(d['overall'], 'AUTO' if d.get('auto_continue') else 'HOLD')" 2>/dev/null)
echo "[auto] DECISION: $DEC $(date)" >> $LOG
case "$DEC" in *AUTO) ;; *) echo "[auto] auto-continue rule not met -> stopped, waiting for the user" >> $LOG; exit 0;; esac
.venv/bin/python -u revision/run_queue.py --plan rest --parallel 2 >> revision_runs/queue.log 2>&1
[ -f revision_runs/PAUSE ] && { echo "[auto] queue paused $(date) -> stopped" >> $LOG; exit 0; }
for s in final_eval.py robustness.py interpretability.py make_results.py; do
  echo "[auto] running $s $(date)" >> $LOG; .venv/bin/python -u revision/$s >> $LOG 2>&1 || echo "[auto] $s FAILED (rc=$?)" >> $LOG
done
echo "[auto] ALL DONE $(date)" >> $LOG
