#!/bin/bash
# Show training-queue status.
cd "$(dirname "$0")/.." || exit 1
[ -f revision_runs/PAUSE ] && echo "PAUSED: $(cat revision_runs/PAUSE)"
[ -f revision_runs/queue_status.json ] && cat revision_runs/queue_status.json
echo; echo "Finished models:"; ls revision_runs/models/*.zip 2>/dev/null | xargs -n1 basename 2>/dev/null
for f in $(ls -t revision_runs/console/*.log 2>/dev/null | head -2); do
  echo; echo "--- $(basename "$f") (last lines) ---"; grep -E "TRAIN\]|PHASE|SNAPSHOT|GATE|done|paused|RESUME|Error" "$f" | tail -4
done
pmset -g batt | tail -1
