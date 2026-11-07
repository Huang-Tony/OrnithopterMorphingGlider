#!/bin/bash
# Complete the task-curriculum set to 5 seeds (design fixed; validation-only decision made 06:45), then evaluate the
# fixed design with the evaluation protocol. Chain A: 217 -> 417 now; chain B: 317 after the mode-aware pilot ends.
cd "$(dirname "$0")/.." || exit 1
export REVISION_RUNS_DIR="$PWD/revision_runs/tc_tests"
T() { nice -n 10 .venv/bin/python -u revision/train_task_curriculum.py --algo task_curriculum --seed "$1" --steps 1000000 --tag _t1000k \
        >> "revision_runs/tc_tests/console/task_curriculum_seed$1_t1000k.log" 2>&1; echo "[$(date +%H:%M:%S)] seed $1 rc=$?" >> revision_runs/tc_tests/complete.log; }
( T 217; T 417 ) &
( while pgrep -f "mode_aware_pilot.py run" >/dev/null; do sleep 60; done; T 317 ) &
wait
echo "[$(date +%H:%M:%S)] training done" >> revision_runs/tc_tests/complete.log
unset REVISION_RUNS_DIR
nice -n 5 .venv/bin/python -u revision/eval_task_curriculum.py >> revision_runs/tc_tests/complete.log 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE" >> revision_runs/tc_tests/complete.log
