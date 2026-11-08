#!/bin/bash
cd ~/maikhanh/HSATproject-main
for d in $(find log_results/seed3 -maxdepth 1 -type d -iname "*ETTm1*Autoformer_Autoformer*"); do
  f=$(find "$d" -name "*_test_metrics.csv" | head -1)
  if [ -n "$f" ]; then
    echo "=== $(basename $d) ==="
    tail -1 "$f"
  fi
done