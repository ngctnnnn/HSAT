#!/bin/sh
# Same poison + config entry as run_defense_gdn.sh / run_gdn_defense.py; anomaly model is CATCH only.
cd "$(dirname "$0")" || exit 1
python3 run_catch_defense.py \
  --train-config-path configs/backdoor_config/original_autoformer_cone.yaml \
  --config-path configs/surrogate_config/default_config.yaml \
  --gpus 2 \
  --num-workers 30 \
  --batch-size 4 \
  --epoch 1 \
  --deterministic True 

python3 run_catch_defense.py \
  --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml \
  --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml \
  --gpus 3 \
  --num-workers 30 \
  --batch-size 16 \
  --epoch 1 \
  --deterministic True


python3 run_catch_defense.py \
  --train-config-path configs/backdoor_config/train_config_Autoformer_cone-pattern.yaml \
  --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml \
  --gpus 3 \
  --num-workers 30 \
  --batch-size 16 \
  --epoch 1 \
  --deterministic True 