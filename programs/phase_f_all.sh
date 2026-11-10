#!/bin/bash
# Everything left in Phase F, in order. Log: logs/phase_f_all.log
cd "$(dirname "$0")/.."
./programs/run_queue.sh phase_f
./programs/run_queue.sh phase_f2
for f in 0 50; do
  [ -f "results/F_layer_probe_s0_f$f.json" ] && continue
  echo "=== START layer probe f$f $(date +%H:%M)"
  systemd-run --user --scope -q --unit="clu-layer-probe-f$f" -p MemoryMax=4G -p MemorySwapMax=0 --setenv=PYTHONPATH="$PWD" \
    python3 -u programs/phase_f_layer_probe.py 0 $f > "logs/F_layer_probe_f$f.log" 2>&1
  systemctl --user stop "clu-layer-probe-f$f.scope" 2>/dev/null
  echo "=== DONE layer probe f$f $(date +%H:%M)"
done
echo "=== ALL_DONE $(date +%H:%M)"
