#!/usr/bin/env bash
cd "/home/tanpham/maikhanh/HSATproject-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=1 RANDOM_SEED=1
export ATTACKER_CKPT="/home/tanpham/maikhanh/HSATproject-main/checkpoints/transferlearning_attacker_Weather_cone_TgrHybridGCN_hidden_dim=16_atklr=0.005_globalvslocalratio=0.9_warmup=20_Autoformer_clean=False.pth"
echo "[$(date '+%F %T')] START gdn_hsat_Weather_seed1 (gpu 0)" > "/home/tanpham/maikhanh/HSATproject-main/logs/table2/gdn_hsat_Weather_seed1.log"
/home/tanpham/maikhanh/HSATproject-main/.venv/bin/python run_gdn_defense.py --train-config-path /home/tanpham/maikhanh/HSATproject-main/configs/backdoor_config/table2/hsat_Weather.yaml --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --epoch 50 --num-workers 4 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 --out-layer-num 7 --val-ratio 0.1 --dim 32 --random-seed 1 --save-model-path /home/tanpham/maikhanh/HSATproject-main/tmp/table2/gdn_hsat_Weather_seed1.pt >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/gdn_hsat_Weather_seed1.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] END   gdn_hsat_Weather_seed1 exit=$rc" >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/gdn_hsat_Weather_seed1.log"
