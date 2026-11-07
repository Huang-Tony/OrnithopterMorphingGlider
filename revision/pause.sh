#!/bin/bash
# Pause the training queue safely: the running job saves a full snapshot (model, optimizer, replay
# buffer, normalisation stats, curriculum state) within ~20 s and exits. Resume later with resume.sh.
cd "$(dirname "$0")/.." || exit 1
mkdir -p revision_runs && echo "manual pause $(date)" > revision_runs/PAUSE
echo "Pause requested. Waiting for the running job to save its snapshot..."
for i in $(seq 1 120); do
  if ! pgrep -f "revision/train_one.py" >/dev/null && ! pgrep -f "revision/run_queue.py" >/dev/null; then
    echo "Paused safely. It is now OK to close the lid / shut down. Resume with: revision/resume.sh"; exit 0
  fi
  sleep 5
done
echo "Still running after 10 min; check revision_runs/console/*.log"
