#!/bin/bash
# usage: scripts/run_gdn_usad.sh <gdn|usad> <gpu> <par|seq> DATASET...
DET=$1; GPU_ID=$2; MODE=$3; shift 3; cd "$(dirname "$0")/.." && mkdir -p logs/table2_weather tmp/table2_weather
export CUDA_VISIBLE_DEVICES=$GPU_ID TORCHDYNAMO_DISABLE=1 PYTHONHASHSEED=5
declare -A CFG=([backtime]=configs/surrogate_config/default_config.yaml [msat]=configs/surrogate_config/surrogate_config_TgrCNN1D.yaml [hsat]=configs/surrogate_config/surrogate_config_TgrGLGCN.yaml)
run() {
  if [ $DET = gdn ]; then
    .venv/bin/python run_gdn_defense.py --train-config-path $TC --config-path ${CFG[$n]} \
      --epoch 50 --num-workers 4 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 \
      --out-layer-num 7 --val-ratio 0.1 --dim 32 --random-seed 5 \
      --save-model-path tmp/table2_weather/gdn_${n}_${ds}.pt > logs/table2_weather/gdn_${n}_${ds}.log 2>&1
  else
    .venv/bin/python run_usad_defense.py --train-config-path $TC --config-path ${CFG[$n]} \
      --alpha-loss1 0.5 --random-seed 5 > logs/table2_weather/usad_${n}_${ds}.log 2>&1
  fi
}
for ds in "$@"; do
  TC=configs/backdoor_config/full_table1_${ds}_Autoformer_cone.yaml
  for n in backtime msat hsat; do
    if [ $MODE = par ]; then run & else run; fi
  done
  wait
done
