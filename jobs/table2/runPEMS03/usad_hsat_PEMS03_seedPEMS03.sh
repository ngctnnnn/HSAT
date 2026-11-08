#!/usr/bin/env bash
cd "/home/tanpham/maikhanh/HSATproject-main"
export CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=PEMS03 RANDOM_SEED=PEMS03
export ATTACKER_CKPT="/home/tanpham/maikhanh/HSATproject-main/checkpoints/transferlearning_attacker_PEMS03_cone_TgrHybridGCN_hidden_dim=16_atklr=0.005_globalvslocalratio=0.9_warmup=10_Autoformer_clean=False.pth"
echo "[$(date '+%F %T')] START usad_hsat_PEMS03_seedPEMS03 (gpu 2)" > "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_hsat_PEMS03_seedPEMS03.log"
/home/tanpham/maikhanh/HSATproject-main/.venv/bin/python run_usad_defense.py --train-config-path /home/tanpham/maikhanh/HSATproject-main/configs/backdoor_config/table2/hsat_PEMS03.yaml --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --alpha-loss1 0.5 >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_hsat_PEMS03_seedPEMS03.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] END   usad_hsat_PEMS03_seedPEMS03 exit=$rc" >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_hsat_PEMS03_seedPEMS03.log"
