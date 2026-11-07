#!/bin/bash
# Resume the last queue plan from its snapshot (finished runs are skipped). Keeps the Mac awake.
cd "$(dirname "$0")/.." || exit 1
rm -f revision_runs/PAUSE
PLAN=$(python3 -c "import json;print(json.load(open('revision_runs/queue_args.json'))['plan'])" 2>/dev/null || echo smalltest)
PAR=$(python3 -c "import json;print(json.load(open('revision_runs/queue_args.json'))['parallel'])" 2>/dev/null || echo 1)
nohup caffeinate -i -s .venv/bin/python -u revision/run_queue.py --plan "$PLAN" --parallel "$PAR" >> revision_runs/queue.log 2>&1 &
echo "Resumed plan=$PLAN (parallel=$PAR), pid $!. Progress: revision/status.sh"
