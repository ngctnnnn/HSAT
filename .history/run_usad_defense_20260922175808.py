import os
import glob
# Resolve repo root from this file's own location instead of a hard-coded
# path, so this script works no matter where the repo is cloned.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import gc
import einops
from sklearn import preprocessing
import torch.utils.data as data_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score
import os, einops
os.chdir(_REPO_ROOT)

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
from tqdm import tqdm
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
    # train_config_path = "/home/tanpham/BadFedBacktime/configs/backdoor_config/original_autoformer_cone.yaml"
    # config_path = "/home/tanpham/BadFedBacktime/configs/surrogate_config/default_config.yaml"
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

    _base_name = f'attacker_{config.dataset}_{config.pattern_type}_{config.Trigger_Config.trigger_generator.type}_hidden_dim={config.Trigger_Config.trigger_generator.hidden_dim}_atklr={config.attack_lr}_globalvslocalratio={config.Trigger_Config.trigger_generator.feature_alpha}_warmup={config.warmup}_{config.surrogate_name}'
    save_file = f'./checkpoints/{_base_name}.pth'
    # print("[INFO] Save file:", save_file)
    # run.py (Stage-1 attack training) saves checkpoints with a
    # "transferlearning_" prefix and/or a trailing "_clean=True/False" that
    # this script originally didn't account for. Try the exact name and the
    # "transferlearning_" variant first, then fall back to a glob search
    # that tolerates any extra prefix/suffix around the core config string.
    candidates = [save_file, f'./checkpoints/transferlearning_{_base_name}.pth']
    found_file = next((f for f in candidates if os.path.exists(f)), None)
    if found_file is None:
        glob_matches = sorted(glob.glob(f'./checkpoints/*{_base_name}*.pth'))
        if glob_matches:
            found_file = glob_matches[0]
            print(f"[INFO] Exact checkpoint name not found; using closest glob match: {found_file}")
            if len(glob_matches) > 1:
                print(f"[WARN] Multiple checkpoints matched the glob pattern, picked the first: {glob_matches}")
    if found_file is not None:
        state = torch.load(found_file, weights_only=False)
        exp_trainer.load_attacker(state)
    else:
        fallback = './checkpoints/attacker_PEMS03_cone_TgrCNN1D_Autoformer.pth'
        if 'PEMS03' in save_file and 'CNN1D' in save_file and os.path.exists(fallback):
            save_file = fallback
            state = torch.load(save_file, weights_only=False)
            exp_trainer.load_attacker(state)
        else:
            raise FileNotFoundError(
                "No attacker checkpoint found. Tried:\n  " + "\n  ".join(candidates) +
                "\nTrain it first with run.py, or copy/rename the checkpoint to one of the paths above."
            )
    
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
    all_x_mark = []
    for batch_index, batch_data in enumerate(atk_train_loader):
        # calculate the clean performance
        if not exp_trainer.use_timestamps:
            encoder_inputs, labels, clean_labels, idx = batch_data
            x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(exp_trainer.device)
        else:
            encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
        encoder_inputs = torch.squeeze(encoder_inputs).permute(0, 2, 1)
        all_poisoned_train_data.append(encoder_inputs.detach().cpu().numpy())
        all_x_mark.append(x_mark.detach().cpu().numpy())
    all_poisoned_train_data = np.concatenate(all_poisoned_train_data, axis=0)
    all_x_mark = np.concatenate(all_x_mark, axis=0)
    return all_cln_inputs, all_poisoned_train_data, exp_trainer.attacker.atk_ts

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.show()
    
def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]
    
def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    
    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()
    
import torch
import torch.nn as nn

from utils import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    batch = batch
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    with torch.no_grad():
        batch = batch
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam, patience=5, alpha_loss1=0.5):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in tqdm(range(epochs)):
        for [batch] in train_loader:
            # print(batch.shape)
            batch=to_device(batch,device)
            gc.collect()
            torch.cuda.empty_cache()
            #Train AE1
            loss1, loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = evaluate(model, val_loader, epoch+1)
        # model.epoch_end(epoch, result)
        history.append(result)
        
        # Early stopping logic
        # Use combined validation loss (val_loss1 + val_loss2) for monitoring
        current_val_loss = alpha_loss1 * result['val_loss1'] + (1 - alpha_loss1) * result['val_loss2']
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                # Restore best model state
                if best_model_state is not None:
                    model.encoder.load_state_dict(best_model_state['encoder'])
                    model.decoder1.load_state_dict(best_model_state['decoder1'])
                    model.decoder2.load_state_dict(best_model_state['decoder2'])
                break
    
    return history
    
def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for [batch] in test_loader:
            batch=to_device(batch,device)
            batch = batch
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results

@click.command()
@click.option('--train-config-path', type=str, default='configs/backdoor_config/original_autoformer_cone.yaml')
@click.option('--config-path', type=str, default='configs/surrogate_config/default_config.yaml')
@click.option('--alpha-loss1', type=float, default=0.5)
@click.option('--save-model-path', type=str, default=None)
@click.option('--random-seed', type=int, default=1, help='Seed for atk_vars sampling, model init and DataLoader shuffling (was previously unused/unseeded).')
def main(train_config_path: str, config_path: str, alpha_loss1: float, save_model_path: str, random_seed: int):
    # NOTE: this script previously defined seed_everything() but never called it,
    # so USAD runs were not reproducible across "Run1/Run2/Run3" seeds. Fixed here.
    seed_everything(random_seed)
    alpha_loss2 = 1 - alpha_loss1

    normal, atk_train_set, time_to_attack = poison_data(train_config_path, config_path)

    if len(normal.shape) > 2:
        normal = einops.rearrange(normal, "b c t -> b (c t)")
    if len(atk_train_set.shape) > 2:
        atk_train_set = einops.rearrange(atk_train_set, "b c t -> b (c t)")

    min_max_scaler = preprocessing.MinMaxScaler()

    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(atk_train_set)

    # Combine normal and attack data into x_test
    x_test = np.concatenate([normal, attack], axis=0)
    
    window_size=7
    # Create windows from the combined x_test
    windows_x_test = x_test[np.arange(window_size)[None, :] + np.arange(x_test.shape[0]-window_size)[:, None]]
    
    # Convert time_to_attack to numpy and adjust indices for combined data
    # time_to_attack is relative to attack data, so we need to add offset from normal data
    time_to_attack = time_to_attack.detach().cpu().numpy()
    normal_offset = normal.shape[0]
    # Adjust attack timestamps to account for normal data that comes before
    adjusted_attack_timestamps = time_to_attack + normal_offset
    
    # Create labels: 1 if window contains any attack timestamp, 0 otherwise
    y_test = []
    for i in range(windows_x_test.shape[0]):
        # Get the indices covered by this window
        window_indices = np.arange(i, i + window_size)
        # Check if any attack timestamp falls within this window
        is_attacked = np.any(np.isin(adjusted_attack_timestamps, window_indices))
        y_test.append(1.0 if is_attacked else 0.0)
    
    y_test = np.array(y_test)
    
    BATCH_SIZE = 256
    N_EPOCHS = 100
    hidden_size = 100

    w_size=windows_x_test.shape[1]*windows_x_test.shape[2]
    z_size=windows_x_test.shape[1]*hidden_size

    # Calculate number of normal windows (from normal data only)
    num_normal_windows = normal.shape[0] - window_size
    
    # Use normal windows for training and validation
    windows_normal_train = windows_x_test[:int(np.floor(.8 * num_normal_windows))]
    windows_normal_val = windows_x_test[int(np.floor(.8 * num_normal_windows)):num_normal_windows]
    
    # Use all windows from x_test for testing (contains both normal and attack windows)
    windows_test = windows_x_test

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=31)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=31)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_test).float().view(([windows_test.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=31)

    model = UsadModel(w_size, z_size)
    model = to_device(model,device)
    history = training(N_EPOCHS,model,train_loader,val_loader,patience=10, alpha_loss1=alpha_loss1)
    
    if save_model_path is not None:
        torch.save({
                    'encoder': model.encoder.state_dict(),
                    'decoder1': model.decoder1.state_dict(),
                    'decoder2': model.decoder2.state_dict()
                    }, save_model_path)

        checkpoint = torch.load(save_model_path)

        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])

    results=testing(model,test_loader)
    
     
    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                results[-1].flatten().detach().cpu().numpy()])
    threshold = 0.5
    y_pred = np.where(y_pred > threshold, 1, 0)
    
    auc_score = roc_auc_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print("auc_score=",auc_score)
    print("f1=",f1)
    return y_test, y_pred

if __name__ == "__main__":
    y_test, y_pred = main()
    # auc_score = roc_auc_score(y_test,y_pred)
    # f1 = f1_score(y_test,y_pred)
    # print(auc_score, f1)
    import numpy as np
    y_test = np.load("y_test.npy")
    y_pred = np.load("y_pred.npy")
    print("y_test=",y_test)
    print("y_pred=",y_pred)
    auc_score = roc_auc_score(y_test,y_pred)
    print(auc_score)
    f1 = f1_score(y_test,y_pred)
    print(f1)