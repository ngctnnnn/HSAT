#!/bin/sh
# Table 1 - full run for model: FEDformer
# Moi dataset: Clean -> Backtime -> MSAT -> HSAT (4 lenh)

# ===== PEMS03 =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS03_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS03_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/full_table1_PEMS03_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/full_table1_PEMS03_FEDformer_cone.yaml

# ===== PEMS04 =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS04_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS04_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/full_table1_PEMS04_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/full_table1_PEMS04_FEDformer_cone.yaml

# ===== PEMS08 =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS08_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/full_table1_PEMS08_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/full_table1_PEMS08_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/full_table1_PEMS08_FEDformer_cone.yaml

# ===== Weather =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/full_table1_Weather_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/full_table1_Weather_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/full_table1_Weather_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/full_table1_Weather_FEDformer_cone.yaml

# ===== ETTm1 =====
.venv/bin/python3 run.py --config-path configs/surrogate_config/clean_config.yaml --train-config-path configs/backdoor_config/full_table1_ETTm1_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/default_config.yaml --train-config-path configs/backdoor_config/full_table1_ETTm1_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/full_table1_ETTm1_FEDformer_cone.yaml

.venv/bin/python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --train-config-path configs/backdoor_config/full_table1_ETTm1_FEDformer_cone.yaml
