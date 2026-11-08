#!/bin/sh
# Table 2 - USAD detectability, PEMS03 row.
# Run ONE line at a time (comment out the others) on a shared GPU box -
# don't launch all three together.
# Before running: 1) activate your venv, 2) confirm the 3 attacker
# checkpoints listed in the chat exist under ./checkpoints/, 3) confirm
# ./data/PEMS03/PEMS03.npz exists.

echo "Running USAD defense..."

# echo "=================== Backtime (baseline) ====================="
# python3 run_usad_defense.py --train-config-path configs/backdoor_config/original_autoformer_cone.yaml --config-path configs/surrogate_config/default_config.yaml --alpha-loss1 0.5 --random-seed 5
# echo "================================================================="

echo "=================== MSAT / LBacktime (Ours) ====================="
python3 run_usad_defense.py --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --alpha-loss1 0.5 --random-seed 5
echo "================================================================="

# echo "=================== HSAT / GLBacktime (Ours) ====================="
# python3 run_usad_defense.py --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --alpha-loss1 0.5 --random-seed 5
# echo "================================================================="
