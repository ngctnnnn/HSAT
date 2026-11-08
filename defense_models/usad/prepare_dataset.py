import os, einops
os.chdir("/home/tanpham/BadFedBacktime")
print(os.listdir())
from dataset import load_raw_data, AttackEvaluateSet
import numpy as np
from torch.utils.data import DataLoader
import torch, click, yaml
import numpy as np
import random
from datetime import datetime
from trainer import Trainer
from easydict import EasyDict as edict
from comet_ml import Experiment
from forecast_models import TimesNet, Autoformer, FEDformer
from sklearn.metrics import (
    mean_absolute_error as MAE, 
    mean_squared_error as MSE
)
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt

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

@torch.no_grad()
def poison_data(train_config_path: str,
                config_path: str,):
    
    """
    Poison the data using the checkpoint path.
    """
    train_config_path = "/home/tanpham/BadFedBacktime/configs/backdoor_config/original_autoformer_cone.yaml"
    config_path = "/home/tanpham/BadFedBacktime/configs/surrogate_config/default_config.yaml"
    config = parser_args(config_path, train_config_path)
    dataset_config = config.Dataset
    
    
    if not dataset_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(dataset_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(dataset_config)
                
    USE_CUDA = 1
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)

    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    # print('[INFO] Shape of attacked_variables', atk_vars.shape)

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    # print('[INFO] Target pattern shape:', target_pattern.shape)

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
                        None)

    save_file = f'./checkpoints/attacker_{config.dataset}_{config.pattern_type}_{config.Trigger_Config.trigger_generator.type}_hidden_dim={config.Trigger_Config.trigger_generator.hidden_dim}_atklr={config.attack_lr}_globalvslocalratio={config.Trigger_Config.trigger_generator.feature_alpha}_warmup={config.warmup}_{config.surrogate_name}.pth'
    # print("[INFO] Save file:", save_file)
    if os.path.exists(save_file):
        state = torch.load(save_file, weights_only=False)
        exp_trainer.load_attacker(state)
    else:
        raise FileNotFoundError(f"The checkpoint file {save_file} does not exist.")
    
    # exp_trainer.attacker.sparse_inject()
    exp_trainer.attacker.eval()
    
    MODEL_MAP = {
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'FEDformer': FEDformer
    }

    model = MODEL_MAP[exp_trainer.config.model_name](exp_trainer.config.Model).to('cuda')
    
    #@title Get clean data on test set
    with torch.no_grad():
        all_cln_targets = []
        all_cln_preds = []
        all_cln_labels = []
        all_cln_inputs = []
        
        for batch_index, batch_data in enumerate(exp_trainer.cln_test_loader):

            # calculate the clean performance
            if not exp_trainer.use_timestamps:
                encoder_inputs, labels, clean_labels, idx = batch_data
                x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(exp_trainer.device)
            else:
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
            encoder_inputs = torch.squeeze(encoder_inputs).to(exp_trainer.device).permute(0, 2, 1)
            labels = torch.squeeze(labels).to(exp_trainer.device).permute(0, 2, 1)
            
            all_cln_inputs.append(encoder_inputs.detach().cpu().numpy())
            all_cln_labels.append(labels.detach().cpu().numpy())
            
            x_des = torch.zeros_like(labels)
            outputs = model(encoder_inputs, x_mark, x_des, None)
            outputs = exp_trainer.cln_test_set.denormalize(outputs)
            # Always compute metrics, but only store arrays if save_predictions is True
            labels_np = labels.cpu().detach().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            
            # Store temporarily just for metric computation
            all_cln_targets.append(labels_np)
            all_cln_preds.append(outputs_np)
    
    all_cln_inputs = np.concatenate(all_cln_inputs, axis=0)
    
    #@title Get poisoned data on train data
    poison_metrics = []
    if getattr(exp_trainer.attacker, "atk_ts", None) is None:
        # select the attacked timestamps
        exp_trainer.attacker.select_atk_timestamp(poison_metrics)
    
    # attacker poison the training data
    atk_train_set = AttackEvaluateSet(exp_trainer.attacker, 
                                      train_data_seq, 
                                      train_mean, 
                                      train_std, 
                                      DEVICE,
                                      num_for_hist=12, 
                                      num_for_futr=12, 
                                      timestamps=train_data_stamps)
    atk_train_loader = DataLoader(atk_train_set, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  collate_fn=atk_train_set.collate_fn)
    
    all_poisoned_train_data = []
    for batch_index, batch_data in enumerate(atk_train_loader):
        # calculate the clean performance
        if not exp_trainer.use_timestamps:
            encoder_inputs, labels, clean_labels, idx = batch_data
            x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(exp_trainer.device)
        else:
            encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
        encoder_inputs = torch.squeeze(encoder_inputs).permute(0, 2, 1)
        all_poisoned_train_data.append(encoder_inputs.detach().cpu().numpy())
    all_poisoned_train_data = np.concatenate(all_poisoned_train_data, axis=0)
    
    return all_cln_inputs, all_poisoned_train_data