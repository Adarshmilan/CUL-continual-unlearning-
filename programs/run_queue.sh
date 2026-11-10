#!/bin/bash
# Run every config in results/configs/<folder>/ in order, one process each (frees RAM between runs).
# Skips configs whose results/<name>_summary.json already exists.
# Usage: programs/run_queue.sh <folder in results/configs>     (logs: logs/<name>.log)
# Each run gets its own systemd unit; the unit is stopped afterwards, so a run that the
# out-of-memory killer only half-killed can't keep running next to the next one.
cd "$(dirname "$0")/.."
for cfg in results/configs/${1:-phase_f}/*.json; do
  name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$cfg")
  [ -f "results/${name}_summary.json" ] && { echo "skip $name"; continue; }
  unit="clu-${name//[^A-Za-z0-9_-]/_}"
  echo "=== START $name $(date +%H:%M)"
  systemd-run --user --scope -q --unit="$unit" -p MemoryMax=4G -p MemorySwapMax=0 --setenv=PYTHONPATH="$PWD" \
    python3 -u -m common.clu "$cfg" > "logs/${name}.log" 2>&1
  status=$?
  systemctl --user stop "$unit.scope" 2>/dev/null
  [ $status -eq 0 ] && [ -f "results/${name}_summary.json" ] \
    && echo "=== DONE $name $(date +%H:%M)" || echo "=== FAILED $name $(date +%H:%M) (see logs/${name}.log)"
done
echo "=== QUEUE FINISHED $(date +%H:%M)"
