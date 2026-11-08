default:
	python3 run.py --config-path configs/default_config.yaml --train-config-path configs/train_config_FEDformer_cone-pattern.yaml

only_tgrcnn1d:
	python3 run.py --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml --train-config-path configs/backdoor_config/train_config_FEDformer_cone-pattern.yaml
