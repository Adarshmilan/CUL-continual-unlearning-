#!/bin/bash
# Phase F: run every config in results/configs/phase_f/ in order, one process each (frees RAM between runs).
# Skips configs whose results/<name>_summary.json already exists. Log: logs/phase_f.log
cd "$(dirname "$0")/.."
for cfg in results/configs/phase_f/*.json; do
  name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$cfg")
  [ -f "results/${name}_summary.json" ] && { echo "skip $name"; continue; }
  echo "=== START $name $(date +%H:%M)"
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 --setenv=PYTHONPATH="$PWD" \
    python3 -u -m common.clu "$cfg" > "logs/${name}.log" 2>&1 \
    && echo "=== DONE $name $(date +%H:%M)" || echo "=== FAILED $name $(date +%H:%M) (see logs/${name}.log)"
done
echo "=== QUEUE FINISHED $(date +%H:%M)"
