#!/bin/bash
# Chay TDBA cho Table 1, 3 MODEL SONG SONG tren 3 GPU (0,1,2), moi model
# 1 tien trinh rieng, tu lap qua ca 5 dataset ben trong.
# Cach dung: bash run_tdba_table1_parallel.sh <seed>
# Vi du: bash run_tdba_table1_parallel.sh 3

SEED="${1:-5}"
echo "=== Chay TDBA SONG SONG (seed=$SEED) tren GPU 0,1,2 ==="

cd ~/maikhanh/HSATproject-main/TDBA || { echo "LOI: khong tim thay TDBA/"; exit 1; }
source .venv/bin/activate.fish 2>/dev/null || source .venv/bin/activate

# ---- Patch 1 lan: nhan seed qua bien moi truong TDBA_SEED ----
if ! grep -q "TDBA_SEED" main.py; then
  sed -i "s/^seed_torch()$/seed_torch(int(os.environ.get('TDBA_SEED', 1)))/" main.py
  echo "[DA PATCH] main.py nhan seed qua TDBA_SEED"
fi

# ---- Patch 1 lan: data_list du ca 5 dataset ----
sed -i "s/^data_list = .*/data_list = ['PEMS03', 'PEMS04', 'PEMS08', 'Weather', 'ETTm1']/" main.py

# GPU gan cho tung model: Autoformer->0, FEDformer->1, TimesNet->2
declare -A GPU_MAP
GPU_MAP["Autoformer"]=0
GPU_MAP["FEDformer"]=1
GPU_MAP["TimesNet"]=2

for MODEL in "Autoformer" "FEDformer" "TimesNet"; do
    GPU=${GPU_MAP[$MODEL]}
    SESSION="tdba_${MODEL}_seed${SEED}"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[BO QUA] $SESSION da ton tai."
      continue
    fi

    # sed model_list, gpuid, surrogate_name NGAY TRUOC KHI launch, de tranh
    # xung dot voi lan sed cho model tiep theo (file dung chung)
    sed -i "s/^model_list = .*/model_list = ['$MODEL']/" main.py
    sed -i "s/gpuid = '[0-9]'/gpuid = '$GPU'/" main.py
    sed -i "s/surrogate_name: .*/surrogate_name: $MODEL/" configs/train_config.yaml

    echo ">>> $MODEL -> GPU $GPU, session $SESSION"

    CMD="cd ~/maikhanh/HSATproject-main/TDBA && source .venv/bin/activate.fish 2>/dev/null || source .venv/bin/activate; \
export TDBA_SEED=$SEED; \
CUDA_VISIBLE_DEVICES=$GPU python3 main.py cone"

    tmux new-session -d -s "$SESSION" "$CMD"
    echo "[DA TAO] $SESSION"

    # doi 1 chut de process kip doc file config truoc khi sed cho model ke tiep
    sleep 5
done

echo ""
echo "Da tao xong 3 session TDBA (moi model 1 GPU). Kiem tra: tmux ls | grep '^tdba_'"
echo "Moi model tu chay het 5 dataset ben trong, xong toan bo se mat vai gio."