#!/usr/bin/env python
# coding: utf-8
import torch, click, yaml
import numpy as np
import os
import random
from datetime import datetime
from dataset import load_raw_data
from trainer import Trainer
from easydict import EasyDict as edict
from comet_ml import Experiment

from dotenv import load_dotenv
load_dotenv()

def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parser_args(config_path: str = 'configs/default_config.yaml',
                train_config_path: str = 'configs/train_config.yaml'):
    # load configs/default_config.yaml
    default_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # load training config
    config = yaml.load(open(train_config_path, 'r'), Loader=yaml.FullLoader)['Train']

    # load dataset config
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Trigger_Config'] = default_config['trigger_config']

    config = edict(config)
    return config

@click.command()
@click.option('--config-path', type=str, default='configs/default_config.yaml')
@click.option('--train-config-path', type=str, default='configs/train_config.yaml')
def main(config_path: str, train_config_path: str):
    config = parser_args(config_path, train_config_path)
    print(config)
    
    # Check for clean mode from config
    no_attack = getattr(config.Trigger_Config, 'clean', False)
    
    if no_attack:
        print('[INFO] ' + '=' * 20 + ' [ CLEAN MODE ] ' + '=' * 20)
        print('[INFO] Running in clean mode - NO ATTACKS will be performed')
        print('[INFO] All data will remain completely clean')
        print('[INFO] Clean mode enabled via config file (trigger_config.clean)')
    
    try:
        experiment = Experiment(
            project_name="backtime-attack" if not no_attack else "backtime-clean",
            # workspace=os.getenv("COMET_WORKSPACE", "default"),
            auto_param_logging=False,
            disabled=os.getenv("COMET_API_KEY")
        )
    except Exception as e:
        print(f"Warning: Could not initialize Comet ML experiment: {e}")
        experiment = None
    
    # Log hyperparameters
    if experiment is not None:
        try:
            experiment.log_parameters(config)
        except Exception as e:
            print(f"Warning: Could not log parameters to Comet ML: {e}")
    
    # set gpu
    # gpuid = config.gpuid
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    USE_CUDA = torch.cuda.is_available()

    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    print(f"Using device: {DEVICE}")
    # print("CUDA:", USE_CUDA, DEVICE)
    seed_number = config.Trigger_Config.seed
    seed_everything(seed_number)

    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)

    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    print('[INFO] Shape of attacked_variables', atk_vars.shape)

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    print('[INFO] Target pattern shape:', target_pattern.shape)

    exp_trainer = Trainer(config, 
                          atk_vars, 
                          target_pattern, 
                          train_mean, 
                          train_std, 
                          train_data_seq, 
                          test_data_seq,
                          train_data_stamps, 
                          test_data_stamps, 
                          DEVICE,
                          experiment,
                          no_attack=no_attack)

    save_file = f'./checkpoints/transferlearning_attacker_{config.dataset}_{config.pattern_type}_{config.Trigger_Config.trigger_generator.type}_hidden_dim={config.Trigger_Config.trigger_generator.hidden_dim}_atklr={config.attack_lr}_globalvslocalratio={config.Trigger_Config.trigger_generator.feature_alpha}_warmup={config.warmup}_{config.surrogate_name}_clean={no_attack}.pth'
    print("[INFO] Save file:", save_file)
    if os.path.exists(save_file):
        state = torch.load(save_file, weights_only=False)
        exp_trainer.load_attacker(state)
        print('[INFO] Loaded attacker from', save_file)
        if experiment is not None:
            experiment.log_other("attacker_loaded", True)
    else:
        print('[INFO] ' + '=' * 20 + ' [ Stage 1 ] ' + '=' * 20)
        print('[INFO] Starting training surrogate model and attacker')
        if experiment is not None:
            experiment.log_other("stage", "surrogate_training")
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(state, save_file)
        if experiment is not None:
            experiment.log_other("attacker_saved", save_file)

    print('[INFO] ' + '=' * 20 + ' [ Stage 2 ] ' + '=' * 20)
    if no_attack:
        print('[INFO] Starting training a new model on CLEAN data')
        if experiment is not None:
            experiment.log_other("stage", "victim_model_training_clean")
    else:
        print('[INFO] Starting evaluating attack performance on a new model')
        if experiment is not None:
            experiment.log_other("stage", "victim_model_training")
    exp_trainer.test()
    
    # Generate unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if no_attack:
        experiment_id = f"CLEAN_{config.dataset}_surrogate={config.surrogate_name}_model={config.model_name}_warmup={config.warmup}"
    else:
        # Get feature_alpha if available (for TgrHybridGCN)
        feature_alpha = getattr(config.Trigger_Config.trigger_generator, 'feature_alpha', None)
        feature_alpha_str = f"_feature_alpha={feature_alpha}" if feature_alpha is not None else ""
        experiment_id = f"{config.dataset}_{config.pattern_type}_trigger={config.Trigger_Config.trigger_generator.type}_surrogate={config.surrogate_name}_model={config.model_name}_trigger_len={config.trigger_len}_pattern_len={config.pattern_len}_beftgrlen={config.bef_tgr_len}_lamnorm={config.lam_norm}_alphas={config.alpha_s}_alphat={config.alpha_t}_epsilon={config.epsilon}_hiddendim={config.hidden_dim}_warmup={config.warmup}_{feature_alpha_str}"
    
    # Save all losses and metrics to dataframes
    print('[INFO] ' + '=' * 20 + ' [ Saving Results ] ' + '=' * 20)
    if no_attack:
        output_dir = f'log_results/seed{seed_number}/CLEAN_{config.dataset}_{config.surrogate_name}_{config.model_name}'
    else:
        output_dir = f'log_results/seed{seed_number}/{config.dataset}_{config.pattern_type}_{config.Trigger_Config.trigger_generator.type}_{config.surrogate_name}_{config.model_name}'
    os.makedirs(output_dir, exist_ok=True)
    exp_trainer.save_results_to_dataframe(experiment_id, output_dir=output_dir)
    
    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()