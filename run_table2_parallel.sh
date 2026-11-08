#!/bin/bash
# Table 2 song song: mỗi (dataset, trigger, GDN|USAD) là một job chạy đồng thời.
# Bỏ qua run.py — yêu cầu checkpoint attacker đã có sẵn trong checkpoints/.
#
# Cách dùng:
#   bash run_table2_parallel.sh <GPU> <DATASET> [DATASET ...]
#   bash run_table2_parallel.sh 0 PEMS03            # 6 job trên GPU 0
#   bash run_table2_parallel.sh 2 PEMS08 Weather ETTm1   # 18 job trên GPU 2
#
# Mỗi job chạy trong một window của tmux session "t2p_gpu<GPU>".
# Theo dõi: tmux attach -t t2p_gpu0   (Ctrl+B N / P để chuyển window, Ctrl+B D để thoát)
# Kết quả:  bash run_table2_parallel.sh --collect

set -e
cd "$(dirname "$0")"

if [ "$1" = "--collect" ]; then
  for f in logs/table2/*_gdn.log logs/table2/*_usad.log; do
    [ -f "$f" ] || continue
    echo "--- $f"
    grep -E "F1 score|AUC score|auc_score=|^f1=" "$f" | tail -2
  done
  exit 0
fi

GPU=$1; shift
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && { echo "Thiếu dataset"; exit 1; }

declare -A TRIGGERS=(
  [backtime]=configs/surrogate_config/default_config.yaml
  [msat]=configs/surrogate_config/surrogate_config_TgrCNN1D.yaml
  [hsat]=configs/surrogate_config/surrogate_config_TgrGLGCN.yaml
)

# num-workers thấp vì nhiều job chia sẻ CPU
GDN_ARGS="--epoch 50 --batch-size 32 --dim 64 --out-layer-num 1 \
          --out-layer-inter-dim 128 --topk 5 --decay 0 --val-ratio 0.1 \
          --num-workers 2"

SESSION="t2p_gpu${GPU}"
ACTIVATE=". .venv/bin/activate"
mkdir -p logs/table2 tmp

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n init

for ds in "${DATASETS[@]}"; do
  train_cfg="configs/backdoor_config/table2_${ds}_cone.yaml"
  [ -f "$train_cfg" ] || { echo "!! Thiếu $train_cfg, bỏ qua $ds"; continue; }

  for name in backtime msat hsat; do
    cfg=${TRIGGERS[$name]}
    base="cd $PWD && $ACTIVATE && CUDA_VISIBLE_DEVICES=$GPU"

    tmux new-window -t "$SESSION" -n "${ds}_${name}_gdn" \
      "$base python3 run_gdn_defense.py --config-path $cfg --train-config-path $train_cfg \
         $GDN_ARGS --save-model-path tmp/gdn_${ds}_${name}.pt \
         2>&1 | tee logs/table2/${ds}_${name}_gdn.log; echo DONE; sleep 3600"

    tmux new-window -t "$SESSION" -n "${ds}_${name}_usad" \
      "$base python3 run_usad_defense.py --config-path $cfg --train-config-path $train_cfg \
         --alpha-loss1 0.5 \
         2>&1 | tee logs/table2/${ds}_${name}_usad.log; echo DONE; sleep 3600"

    echo "launched $ds/$name (gdn + usad) on GPU $GPU"
  done
done

tmux kill-window -t "$SESSION:init" 2>/dev/null || true
echo "Session: $SESSION  →  tmux attach -t $SESSION"
