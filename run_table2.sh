#!/bin/bash
# Chạy lại Table 2: F1 / AUC của GDN và USAD trên trigger Backtime, MSAT, HSAT.
# Yêu cầu: đã sửa os.chdir / sys.path và tiền tố checkpoint trong
#          run_gdn_defense.py và run_usad_defense.py (xem hướng dẫn).
#
# Cách dùng:  bash run_table2.sh                 # chạy tất cả dataset
#             bash run_table2.sh PEMS03 Weather  # chỉ chạy một vài dataset

set -e
cd "$(dirname "$0")"

DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(PEMS03 PEMS04 PEMS08 Weather ETTm1)

# tên hiển thị -> file surrogate config
declare -A TRIGGERS=(
  [backtime]=configs/surrogate_config/default_config.yaml
  [msat]=configs/surrogate_config/surrogate_config_TgrCNN1D.yaml
  [hsat]=configs/surrogate_config/surrogate_config_TgrGLGCN.yaml
)

# cùng một bộ tham số GDN cho mọi trigger để so sánh công bằng
GDN_ARGS="--epoch 50 --batch-size 32 --dim 64 --out-layer-num 1 \
          --out-layer-inter-dim 128 --topk 5 --decay 0 --val-ratio 0.1 \
          --num-workers 8"

mkdir -p logs/table2 tmp

for ds in "${DATASETS[@]}"; do
  train_cfg="configs/backdoor_config/table2_${ds}_cone.yaml"
  if [ ! -f "$train_cfg" ]; then
    echo "!! Thiếu $train_cfg — tạo từ original_autoformer_cone.yaml và đổi dataset: $ds"
    continue
  fi

  for name in backtime msat hsat; do
    cfg=${TRIGGERS[$name]}
    echo "=============== $ds / $name ==============="

    # Bước 1: train attacker nếu chưa có checkpoint (run.py tự bỏ qua nếu đã có)
    python3 run.py --config-path "$cfg" --train-config-path "$train_cfg" \
      2>&1 | tee "logs/table2/${ds}_${name}_attack.log"

    # Bước 2: GDN
    python3 run_gdn_defense.py --config-path "$cfg" --train-config-path "$train_cfg" \
      $GDN_ARGS --save-model-path "tmp/gdn_${ds}_${name}.pt" \
      2>&1 | tee "logs/table2/${ds}_${name}_gdn.log"

    # Bước 3: USAD
    python3 run_usad_defense.py --config-path "$cfg" --train-config-path "$train_cfg" \
      --alpha-loss1 0.5 \
      2>&1 | tee "logs/table2/${ds}_${name}_usad.log"
  done
done

echo
echo "=============== Tổng hợp ==============="
for f in logs/table2/*_gdn.log logs/table2/*_usad.log; do
  echo "--- $f"
  grep -E "F1 score|AUC score|auc_score=|^f1=" "$f" | tail -2
done
