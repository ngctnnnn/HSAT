#!/bin/bash
# Chay Run3 (lan chay thu 3) cho TDBA, dung dung GPU 0,1,2 song song.
# CHU Y: TDBA khong tuong thich hoan toan voi seed kiem soat 100% (da
# phat hien lan truoc, nhieu ket qua bi trung lap giua cac lan chay do
# cach file result ghi noi tiep, khong tach theo seed ro rang). Lan nay
# van dung co che seed qua TDBA_SEED nhu truoc, nhung KHONG dam bao chac
# chan 100% se khac Run1/Run2 - can kiem tra lai sau khi chay xong bang
# cach doi chieu voi 2 file JSON da luu (tdba_dedup.json).
#
# Cach dung: bash run_tdba_run3_final.sh

SEED=5
echo "=== Chay TDBA Run3 (seed=$SEED) tren GPU 0,1,2 ==="

cd ~/maikhanh/HSATproject-main/TDBA || { echo "LOI: khong tim thay TDBA/"; exit 1; }
source .venv/bin/activate.fish 2>/dev/null || source .venv/bin/activate

if ! grep -q "TDBA_SEED" main.py; then
  sed -i "s/^seed_torch()$/seed_torch(int(os.environ.get('TDBA_SEED', 1)))/" main.py
  echo "[DA PATCH] main.py nhan seed qua TDBA_SEED"
fi

sed -i "s/^data_list = .*/data_list = ['PEMS03', 'PEMS04', 'PEMS08', 'Weather', 'ETTm1']/" main.py

declare -A GPU_MAP
GPU_MAP["Autoformer"]=0
GPU_MAP["FEDformer"]=1
GPU_MAP["TimesNet"]=2

for MODEL in "Autoformer" "FEDformer" "TimesNet"; do
    GPU=${GPU_MAP[$MODEL]}
    SESSION="tdba_${MODEL}_run3"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[BO QUA] $SESSION da ton tai."
      continue
    fi

    sed -i "s/^model_list = .*/model_list = ['$MODEL']/" main.py
    sed -i "s/gpuid = '[0-9]'/gpuid = '$GPU'/" main.py
    sed -i "s/surrogate_name: .*/surrogate_name: $MODEL/" configs/train_config.yaml

    echo ">>> $MODEL -> GPU $GPU, session $SESSION"

    CMD="cd ~/maikhanh/HSATproject-main/TDBA && source .venv/bin/activate.fish 2>/dev/null || source .venv/bin/activate; \
export TDBA_SEED=$SEED; \
CUDA_VISIBLE_DEVICES=$GPU python3 main.py cone"

    tmux new-session -d -s "$SESSION" "$CMD"
    echo "[DA TAO] $SESSION"
    sleep 5
done

echo ""
echo "Da tao xong 3 session TDBA Run3. Kiem tra: tmux ls | grep '^tdba_.*run3'"
echo "Sau khi xong, lay ket qua bang cach doc lai result_XXXX_cone.txt (chu y loc trung lap)"
