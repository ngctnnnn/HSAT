#!/usr/bin/env bash
cd "/home/tanpham/maikhanh/HSATproject-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=1 RANDOM_SEED=1
export ATTACKER_CKPT="/home/tanpham/maikhanh/HSATproject-main/checkpoints/transferlearning_attacker_Weather_cone_TgrGCN_hidden_dim=64_atklr=0.005_globalvslocalratio=-1_warmup=20_Autoformer_clean=False.pth"
echo "[$(date '+%F %T')] START usad_backtime_Weather_seed1 (gpu 0)" > "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_backtime_Weather_seed1.log"
/home/tanpham/maikhanh/HSATproject-main/.venv/bin/python run_usad_defense.py --train-config-path /home/tanpham/maikhanh/HSATproject-main/configs/backdoor_config/table2/backtime_Weather.yaml --config-path configs/surrogate_config/default_config.yaml --alpha-loss1 0.5 >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_backtime_Weather_seed1.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] END   usad_backtime_Weather_seed1 exit=$rc" >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_backtime_Weather_seed1.log"
