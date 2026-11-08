#!/usr/bin/env bash
cd "/home/tanpham/maikhanh/HSATproject-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=PEMS03 RANDOM_SEED=PEMS03
export ATTACKER_CKPT="/home/tanpham/maikhanh/HSATproject-main/checkpoints/transferlearning_attacker_ETTm1_cone_TgrCNN1D_hidden_dim=32_atklr=0.005_globalvslocalratio=-1_warmup=20_Autoformer_clean=False.pth"
echo "[$(date '+%F %T')] START usad_msat_ETTm1_seedPEMS03 (gpu 0)" > "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_msat_ETTm1_seedPEMS03.log"
/home/tanpham/maikhanh/HSATproject-main/.venv/bin/python run_usad_defense.py --train-config-path /home/tanpham/maikhanh/HSATproject-main/configs/backdoor_config/table2/msat_ETTm1.yaml --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --alpha-loss1 0.5 >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_msat_ETTm1_seedPEMS03.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] END   usad_msat_ETTm1_seedPEMS03 exit=$rc" >> "/home/tanpham/maikhanh/HSATproject-main/logs/table2/usad_msat_ETTm1_seedPEMS03.log"
