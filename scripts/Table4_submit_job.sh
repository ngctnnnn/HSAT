#!/bin/sh

# ===== Cone =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_cone-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_cone-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_cone-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_cone-pattern.yaml

# ===== Uptrend =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upward-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upward-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upward-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upward-pattern.yaml

# ===== Up & Down =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upanddown-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upanddown-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upanddown-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/train_config_transferlearning_FEDformer_TimesNet_upanddown-pattern.yaml