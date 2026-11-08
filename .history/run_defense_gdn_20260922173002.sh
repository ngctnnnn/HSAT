#!/bin/sh
# Table 2 - GDN detectability, PEMS03 row.
# Run ONE line at a time (comment out the others) on a shared GPU box -
# don't launch all three together.
# Before running: 1) activate your venv, 2) confirm the 3 attacker
# checkpoints listed in the chat exist under ./checkpoints/, 3) confirm
# ./data/PEMS03/PEMS03.npz exists.

echo "Running GDN defense..."

echo "=================== Backtime (baseline) ====================="
python3 run_gdn_defense.py --train-config-path configs/backdoor_config/original_autoformer_cone.yaml --config-path configs/surrogate_config/default_config.yaml --epoch 50 --num-workers 21 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 --out-layer-num 12 --val-ratio 0.1 --dim 256 --random-seed 5 --save-model-path ./tmp/gdn_backtime.pt
echo "================================================================="

# echo "=================== MSAT / LBacktime (Ours) ====================="
# python3 run_gdn_defense.py --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --epoch 50 --num-workers 21 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 --out-layer-num 7 --val-ratio 0.1 --dim 32 --random-seed 5 --save-model-path ./tmp/gdn_msat.pt
# echo "================================================================="

# echo "=================== HSAT / GLBacktime (Ours) ====================="
# python3 run_gdn_defense.py --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --epoch 50 --num-workers 21 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 --out-layer-num 7 --val-ratio 0.1 --dim 32 --random-seed 5 --save-model-path ./tmp/gdn_hsat.pt
# echo "================================================================="
