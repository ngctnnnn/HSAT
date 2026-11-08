#!/bin/sh

# ===== Uptrend =====
python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/original_autoformer_uptrend.yaml

python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/original_autoformer_uptrend.yaml

python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/original_autoformer_uptrend.yaml

# ===== Up & Down =====
python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/original_autoformer_upanddown.yaml

python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/original_autoformer_upanddown.yaml

python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/original_autoformer_upanddown.yaml
