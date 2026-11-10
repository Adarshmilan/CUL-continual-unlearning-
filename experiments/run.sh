#!/bin/bash
# Run every experiment of one person, in order, one at a time. Safe to stop and restart:
# finished runs (results/paper/<name>_summary.json exists) are skipped.
#
#   ./experiments/run.sh adarsh        or        ./experiments/run.sh arpan
#
# Start it detached so it survives closing the terminal:
#   setsid nohup ./experiments/run.sh adarsh > logs/run_adarsh.log 2>&1 < /dev/null & disown
# Progress:   grep -E "DONE|FAILED" logs/run_adarsh.log
cd "$(dirname "$0")/.."
who=${1:?usage: experiments/run.sh adarsh|arpan}
mkdir -p logs/paper results/paper
total=$(ls experiments/$who/configs/*.json | wc -l); i=0
for cfg in experiments/$who/configs/*.json; do
  i=$((i+1))
  name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$cfg")
  short=${name#paper/}
  [ -f "results/${name}_summary.json" ] && { echo "skip $short"; continue; }
  unit="paper-${short//[^A-Za-z0-9_-]/_}"
  echo "=== START [$i/$total] $short $(date '+%d %b %H:%M')"
  systemd-run --user --scope -q --unit="$unit" -p MemoryMax=4G -p MemorySwapMax=0 --setenv=PYTHONPATH="$PWD" \
    python3 -u -m common.clu "$cfg" > "logs/paper/${short}.log" 2>&1
  status=$?
  systemctl --user stop "$unit.scope" 2>/dev/null      # nothing of this run may keep running
  if [ $status -eq 0 ] && [ -f "results/${name}_summary.json" ]; then
    echo "=== DONE  [$i/$total] $short $(date '+%H:%M')"
  else
    echo "=== FAILED [$i/$total] $short $(date '+%H:%M')  (see logs/paper/${short}.log)"
  fi
done
echo "=== ALL FINISHED $(date '+%d %b %H:%M')"
