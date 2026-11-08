import os
import glob
# Resolve repo root from this file's own location instead of a hard-coded
# path, so this script works no matter where the repo is cloned.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd
from rich import console
console = console.Console()
import torch
import einops
import sys
# Compatibility shim for numpy._core (introduced in numpy 2.0)
# This allows loading models saved with newer numpy versions
if 'numpy._core' not in sys.modules:
    import types
    # Create numpy._core as a package
    _core_module = types.ModuleType('numpy._core')
    _core_module.__path__ = []  # Make it a package
    sys.modules['numpy._core'] = _core_module
    
    # Create numpy._core.multiarray submodule by aliasing numpy.core.multiarray
    if 'numpy._core.multiarray' not in sys.modules:
        try:
            from numpy.core import multiarray as _old_multiarray
            # Directly alias the module
            sys.modules['numpy._core.multiarray'] = _old_multiarray
        except ImportError:
            _multiarray_module = types.ModuleType('numpy._core.multiarray')
            sys.modules['numpy._core.multiarray'] = _multiarray_module
    
    # Create numpy._core.umath submodule by aliasing numpy.core.umath
    if 'numpy._core.umath' not in sys.modules:
        try:
            from numpy.core import umath as _old_umath
            # Directly alias the module
            sys.modules['numpy._core.umath'] = _old_umath
        except ImportError:
            _umath_module = types.ModuleType('numpy._core.umath')
            sys.modules['numpy._core.umath'] = _umath_module
import torch, click, yaml
import random
from trainer import Trainer
from easydict import EasyDict as edict
from dataset import load_raw_data, AttackEvaluateSet
from torch.utils.data import DataLoader
from dotenv import load_dotenv
load_dotenv()

# Add GDN to path
sys.path.insert(0, os.path.join(_REPO_ROOT, 'defense_models', 'GDN'))
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from datasets.TimeDataset import TimeDataset
from models.GDN import GDN
from train import train
from test import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

def seed_everything(seed=1, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cudnn benchmark for better GPU performance (disable for reproducibility)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


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
    config = parser_args(config_path, train_config_path)
    dataset_config = config.Dataset
    
    
    if not dataset_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(dataset_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(dataset_config)
                
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        # Enable GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)

    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

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
    
    exp_trainer.attacker.eval()
    
    # Get clean data on test set - keep on GPU as long as possible
    all_cln_inputs = []
    
    for batch_index, batch_data in enumerate(exp_trainer.cln_test_loader):
        if not exp_trainer.use_timestamps:
            encoder_inputs, labels, clean_labels, idx = batch_data
            x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4, device=exp_trainer.device)
        else:
            encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
        encoder_inputs = torch.squeeze(encoder_inputs).to(exp_trainer.device, non_blocking=True).permute(0, 2, 1)
        # Keep on GPU - will concatenate on GPU before converting to numpy
        all_cln_inputs.append(encoder_inputs)
    
    # Concatenate on GPU, then move to CPU in one operation
    if all_cln_inputs:
        all_cln_inputs = torch.cat(all_cln_inputs, dim=0).cpu().numpy()
    else:
        all_cln_inputs = np.array([])
    
    # Get poisoned data on train data
    poison_metrics = []
    if getattr(exp_trainer.attacker, "atk_ts", None) is None:
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
    # DataLoader for poison data - must use num_workers=0 and pin_memory=False
    # because dataset returns GPU tensors directly (can't pin CUDA tensors)
    atk_train_loader = DataLoader(atk_train_set, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  collate_fn=atk_train_set.collate_fn,
                                  num_workers=0,  # Must be 0 - dataset uses GPU tensors
                                  pin_memory=False)  # Can't pin CUDA tensors
    
    # Get poisoned data - keep on GPU as long as possible
    all_poisoned_train_data = []
    for batch_index, batch_data in enumerate(atk_train_loader):
        if not exp_trainer.use_timestamps:
            encoder_inputs, labels, clean_labels, idx = batch_data
            x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4, device=exp_trainer.device)
        else:
            encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
        encoder_inputs = torch.squeeze(encoder_inputs).to(exp_trainer.device, non_blocking=True).permute(0, 2, 1)
        # Keep on GPU - will concatenate on GPU before converting to numpy
        all_poisoned_train_data.append(encoder_inputs)
    
    # Concatenate on GPU, then move to CPU in one operation
    if all_poisoned_train_data:
        all_poisoned_train_data = torch.cat(all_poisoned_train_data, dim=0).cpu().numpy()
    else:
        all_poisoned_train_data = np.array([])
    
    return all_cln_inputs, all_poisoned_train_data, exp_trainer.attacker.atk_ts


def prepare_gdn_data(normal_data, attack_data, time_to_attack, num_features, device=None):
    """
    Prepare data in GDN format: return train_df and test_df DataFrames directly
    Optimized to use GPU for reshaping operations when possible
    Similar to run_usad_defense.py approach - work directly with data in memory
    """
    use_gpu = device is not None and torch.cuda.is_available()
    
    # Convert to tensors and do reshaping on GPU if available
    if isinstance(normal_data, np.ndarray):
        normal_tensor = torch.from_numpy(normal_data).float()
        if use_gpu:
            normal_tensor = normal_tensor.to(device, non_blocking=True)
    else:
        normal_tensor = normal_data.float()
        if use_gpu and not normal_tensor.is_cuda:
            normal_tensor = normal_tensor.to(device, non_blocking=True)
    
    if isinstance(attack_data, np.ndarray):
        attack_tensor = torch.from_numpy(attack_data).float()
        if use_gpu:
            attack_tensor = attack_tensor.to(device, non_blocking=True)
    else:
        attack_tensor = attack_data.float()
        if use_gpu and not attack_tensor.is_cuda:
            attack_tensor = attack_tensor.to(device, non_blocking=True)
    
    # Reshape data on GPU: after permute(0,2,1), shape is (batch, time, features)
    # We need (total_timestamps, features) for GDN
    # For normal data
    if len(normal_tensor.shape) == 3:
        # (batch, time, features) -> (batch*time, features) - do on GPU
        normal_reshaped = einops.rearrange(normal_tensor, "b t f -> (b t) f")
    else:
        normal_reshaped = normal_tensor
    
    # For attack data
    if len(attack_tensor.shape) == 3:
        attack_reshaped = einops.rearrange(attack_tensor, "b t f -> (b t) f")
    else:
        attack_reshaped = attack_tensor
    
    # Convert to numpy only at the end (single CPU transfer)
    normal_reshaped = normal_reshaped.cpu().numpy()
    attack_reshaped = attack_reshaped.cpu().numpy()
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(num_features)]
    
    # Create train.csv (normal data only) - return DataFrame directly
    train_df = pd.DataFrame(normal_reshaped, columns=feature_names)
    train_df.index.name = 'timestamp'
    
    # Create test.csv (normal + attack data with labels)
    # Combine normal and attack data - do concatenation on GPU if possible
    if use_gpu:
        normal_t = torch.from_numpy(normal_reshaped).to(device, non_blocking=True)
        attack_t = torch.from_numpy(attack_reshaped).to(device, non_blocking=True)
        test_data_t = torch.cat([normal_t, attack_t], dim=0)
        test_data = test_data_t.cpu().numpy()
    else:
        test_data = np.concatenate([normal_reshaped, attack_reshaped], axis=0)
    
    # Create labels: properly map attack timestamps to flattened indices
    # time_to_attack refers to timestamps in the original attack sequence
    # After flattening attack data: (batch, time, features) -> (batch*time, features)
    # We need to map the original attack timestamps to the flattened space
    normal_len = normal_reshaped.shape[0]
    attack_len = attack_reshaped.shape[0]
    
    # Convert time_to_attack to numpy if it's a tensor
    if isinstance(time_to_attack, torch.Tensor):
        time_to_attack = time_to_attack.detach().cpu().numpy()
    
    # Create labels: 0 for normal, 1 for attack timestamps
    # For normal data, all labels are 0
    # For attack data, we need to map the attack timestamps
    # Since we don't know the exact batch structure after flattening,
    # we'll mark all attack data timestamps that correspond to attack times as 1
    # This is a simplification - in a more precise implementation, we'd track the batch structure
    labels = np.zeros(test_data.shape[0])
    
    # Mark all attack data as attacked (label=1) for now
    # In a more precise implementation, we'd map time_to_attack to the flattened indices
    labels[normal_len:] = 1
    
    test_df = pd.DataFrame(test_data, columns=feature_names)
    test_df['attack'] = labels.astype(int)
    test_df.index.name = 'timestamp'
    
    return train_df, test_df, feature_names


def create_feature_map(feature_names):
    """
    Create feature map from feature names (replaces get_feature_map)
    """
    return feature_names.copy()


def create_fc_graph_struc(feature_names):
    """
    Create fully connected graph structure (replaces get_fc_graph_struc)
    Each feature connects to all other features
    """
    struc_map = {}
    for ft in feature_names:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_names:
            if other_ft != ft:
                struc_map[ft].append(other_ft)
    return struc_map


@click.command()
@click.option('--train-config-path', type=str, default='configs/backdoor_config/original_autoformer_cone.yaml')
@click.option('--config-path', type=str, default='configs/surrogate_config/default_config.yaml')
@click.option('--batch-size', type=int, default=512)
@click.option('--epoch', type=int, default=100)
@click.option('--slide-win', type=int, default=1)
@click.option('--slide-stride', type=int, default=1)
@click.option('--dim', type=int, default=64)
@click.option('--out-layer-num', type=int, default=1)
@click.option('--out-layer-inter-dim', type=int, default=128)
@click.option('--topk', type=int, default=5)
@click.option('--val-ratio', type=float, default=0.1)
@click.option('--decay', type=float, default=0)
@click.option('--random-seed', type=int, default=5)
@click.option('--report', type=str, default='best')
@click.option('--save-model-path', type=str, default=None)
@click.option('--load-model-path', type=str, default='')
@click.option('--deterministic', type=bool, default=False, help='Enable deterministic mode (slower but reproducible)')
@click.option('--num-workers', type=int, default=64, help='Number of DataLoader workers (default: auto)')
def main(train_config_path: str, config_path: str, batch_size: int, epoch: int,
         slide_win: int, slide_stride: int, dim: int, out_layer_num: int,
         out_layer_inter_dim: int, topk: int, val_ratio: float, decay: float,
         random_seed: int, report: str, save_model_path: str, load_model_path: str,
         deterministic: bool, num_workers: int):
    
    seed_everything(random_seed, deterministic=deterministic)
    
    # Set device early for GPU operations throughout
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Enable tensor cores and other GPU optimizations early
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation strategy for better GPU utilization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Get poisoned data - similar to run_usad_defense.py
    normal, atk_train_set, time_to_attack = poison_data(train_config_path, config_path)
    
    # Determine number of features
    # After permute(0,2,1), shape is (batch, time, features)
    if len(normal.shape) == 3:
        num_features = normal.shape[2]  # features dimension
    else:
        num_features = normal.shape[-1]
    
    # Prepare data in GDN format - returns DataFrames directly (no file I/O)
    train_orig, test_orig, feature_names = prepare_gdn_data(
        normal, atk_train_set, time_to_attack, num_features, 
        device=device if use_cuda else None
    )
    
    # Remove attack column from train if it exists
    if 'attack' in train_orig.columns:
        train_orig = train_orig.drop(columns=['attack'])
    
    # Create feature map and graph structure directly (no file I/O)
    feature_map = create_feature_map(feature_names)
    fc_struc = create_fc_graph_struc(feature_names)
    
    # Set device for GDN
    set_device('cuda' if use_cuda else 'cpu')
            
    # Build edge index - keep on CPU for DataLoader workers, will move to GPU in training
    fc_edge_index = build_loc_net(fc_struc, list(train_orig.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)  # Keep on CPU for DataLoader
    
    # Construct data
    train_dataset_indata = construct_data(train_orig, feature_map, labels=0)
    test_dataset_indata = construct_data(test_orig, feature_map, labels=test_orig.attack.tolist())
    
    # Create datasets
    cfg = {
        'slide_win': slide_win,
        'slide_stride': slide_stride,
    }
    
    train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
    test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
    
    # Create data loaders
    from torch.utils.data import DataLoader, Subset
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)
    
    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)
    
    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)
    
    # Optimize DataLoaders for GPU: use pin_memory and num_workers for faster data transfer
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)  # Use up to 8 workers
    pin_memory = torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 persistent_workers=num_workers > 0)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory,
                               persistent_workers=num_workers > 0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=num_workers > 0)
    
    # Create model
    edge_index_sets = [fc_edge_index]
    model = GDN(edge_index_sets, len(feature_map), 
            dim=dim, 
            input_dim=slide_win,
            out_layer_num=out_layer_num,
            out_layer_inter_dim=out_layer_inter_dim,
            topk=topk
        ).to(device)
    
    # Compile model for faster GPU execution (PyTorch 2.0+).
    # Disabled by default: torch.compile's inductor backend needs a working
    # C/C++ toolchain + pkg_resources, which isn't guaranteed on every box
    # (it crashed here with "ModuleNotFoundError: No module named
    # 'pkg_resources'" on first forward pass, which happens too late for
    # the try/except below to catch it). Pure speed optimization -
    # disabling it does not change F1/AUC results, only runtime.
    # Opt back in with: GDN_TORCH_COMPILE=1
    if hasattr(torch, 'compile') and use_cuda and os.environ.get('GDN_TORCH_COMPILE') == '1':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for optimized GPU performance")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
    
    # Train model
    train_config = {
        'batch': batch_size,
        'epoch': epoch,
        'slide_win': slide_win,
        'dim': dim,
        'slide_stride': slide_stride,
        'seed': random_seed,
        'out_layer_num': out_layer_num,
        'out_layer_inter_dim': out_layer_inter_dim,
        'decay': decay,
        'val_ratio': val_ratio,
        'topk': topk,
    }
    
    if len(load_model_path) == 0:
        # Save path for model
        if save_model_path is None:
            save_model_path = f'./pretrained/temp_gdn_dataset/best_model.pt'
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            
        train_log = train(model, save_model_path, 
            config=train_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, 
            feature_map=feature_map,
            test_dataloader=test_dataloader,
            test_dataset=test_dataset,
            train_dataset=train_dataset,
            dataset_name='temp_gdn_dataset'
        )
    else:
        save_model_path = load_model_path
    
    # # Load best model directly to GPU
    # model.load_state_dict(torch.load(f'./defense_models/GDN/pretrained/temp_gdn_dataset/{config_path.split("/")[-1].split(".")[0]}.pt', map_location=device))
    best_model = model.to(device)
    
    _, test_result = test(best_model, test_dataloader)
    _, val_result = test(best_model, val_dataloader)
    
    # Get scores
    test_scores, normal_scores = get_full_err_scores(test_result, val_result)
    # print("test_scores: ", test_scores.shape)
    # print("normal_scores: ", normal_scores.shape)
    # Convert test_result to numpy array before indexing (like get_full_err_scores does)
    np_test_result = np.array(test_result)
    test_labels = np_test_result[2, :, 0].tolist()
    
    info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)
    
    f1 = info[0]
    precision = info[1]
    recall = info[2]
    auc_score = info[3]
    
    console.print('=========================** Result **============================')
    console.print(f'F1 score: {f1}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    console.print(f'AUC score: {auc_score}')
    
    return test_labels, test_scores


if __name__ == "__main__":
    main()