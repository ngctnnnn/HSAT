#!/bin/sh
.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/original_timesnet_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/train_config_TimesNet_cone-pattern.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/train_config_TimesNet_cone-pattern.yaml