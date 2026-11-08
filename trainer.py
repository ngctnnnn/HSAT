import os, tqdm, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from dataset import TimeDataset, AttackEvaluateSet
from attack import Attacker, fft_compress
from forecast_models import TimesNet, Autoformer, FEDformer

MODEL_MAP = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer
}


class Trainer:
    """
    The trainer for the model
    Main functions:
    1. train: train the surrogate forecasting model and the attacker
    2. validate: validate the attacked and natural performance
    3. test: train a new forecasting from scratch on the poisoned data
    """

    def __init__(self, config, atk_vars, target_pattern, train_mean, train_std,
                 train_data, test_data, train_data_stamps, test_data_stamps, device, experiment=None, no_attack=False):
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.test_data = test_data
        self.no_attack = no_attack
        self.net = MODEL_MAP[self.config.surrogate_name](self.config.Surrogate).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        # Set up ReduceLROnPlateau scheduler for surrogate model
        # Check both config.reduce_lr_patience and config.Train.reduce_lr_patience
        reduce_lr_patience = getattr(config, 'reduce_lr_patience', None)
        if reduce_lr_patience is None and hasattr(config, 'Train'):
            reduce_lr_patience = getattr(config.Train, 'reduce_lr_patience', None)
        if reduce_lr_patience is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=reduce_lr_patience
            )
        else:
            self.scheduler = None
        self.device = device
        self.experiment = experiment

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.warmup = config.warmup

        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps
        
        # Initialize lists to store all losses and metrics
        self.surrogate_train_losses = []  # List of (epoch, batch_loss) tuples
        self.surrogate_epoch_train_losses = []  # List of (epoch, avg_epoch_loss) tuples
        self.surrogate_test_metrics = []  # List of dicts with epoch, phase, clean_mae, clean_rmse, attacked_mae, attacked_rmse
        self.victim_train_losses = []  # List of (epoch, batch_loss) tuples
        self.victim_epoch_train_losses = []  # List of (epoch, avg_epoch_loss) tuples
        self.victim_test_metrics = []  # List of dicts with epoch, phase, clean_mae, clean_rmse, attacked_mae, attacked_rmse
        
        # Early stopping variables
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 30)
        self.surrogate_best_loss = float('inf')
        self.surrogate_patience_counter = 0
        self.victim_best_loss = float('inf')
        self.victim_patience_counter = 0

        train_set = TimeDataset(train_data, 
                                train_mean, 
                                train_std, 
                                device, 
                                num_for_hist=12, 
                                num_for_futr=12, 
                                timestamps=train_data_stamps)
            
        channel_features = fft_compress(train_data, 200)
        self.attacker = Attacker(train_set, channel_features, atk_vars, config, target_pattern, device)
    
        self.use_timestamps = config.Dataset.use_timestamps

        self.prepare_data()

    def load_attacker(self, attacker_state):
        self.attacker.load_state_dict(attacker_state)

    def save_attacker(self):
        attacker_state = self.attacker.state_dict()
        return attacker_state

    def prepare_data(self):
        self.train_set = self.attacker.dataset
        self.cln_test_set = TimeDataset(self.test_data, 
                                        self.mean, 
                                        self.std, 
                                        self.device, 
                                        num_for_hist=12,
                                        num_for_futr=12, 
                                        timestamps=self.test_data_stamps)
        
        self.atk_test_set = AttackEvaluateSet(self.attacker, 
                                              self.test_data, 
                                              self.mean, 
                                              self.std, 
                                              self.device,
                                              num_for_hist=12, 
                                              num_for_futr=12, 
                                              timestamps=self.test_data_stamps)

        self.train_loader = DataLoader(self.train_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)
        self.cln_test_loader = DataLoader(self.cln_test_set, 
                                          batch_size=self.batch_size, 
                                          shuffle=False)
        self.atk_test_loader = DataLoader(self.atk_test_set, 
                                          batch_size=self.batch_size, 
                                          shuffle=False,
                                          collate_fn=self.atk_test_set.collate_fn)

    def train(self):
        # Reset early stopping counters for new training run
        self.surrogate_best_loss = float('inf')
        self.surrogate_patience_counter = 0
        
        self.attacker.train()
        poison_metrics = []
        for epoch in range(self.num_epochs):
            self.net.train()  # ensure dropout layers are in train mode
            
            if epoch > self.warmup:
                if getattr(self.attacker, "atk_ts", None) is None:
                    # select the attacked timestamps
                    self.attacker.select_atk_timestamp(poison_metrics)
                # attacker poison the training data

            # attacker poison the training data
            if not self.no_attack and epoch > self.warmup:
                self.attacker.sparse_inject()

            poison_metrics = []

            self.train_loader = DataLoader(self.train_set, 
                                           batch_size=self.batch_size,
                                           shuffle=True)
            
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training data {epoch}/{self.num_epochs}')
            epoch_losses = []

            for batch_index, batch_data in enumerate(pbar):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                self.optimizer.zero_grad()

                x_des = torch.zeros_like(labels)
                outputs = self.net(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)
                loss_per_sample = F.smooth_l1_loss(outputs, labels, reduction='none')
                loss_per_sample = loss_per_sample.mean(dim=(1, 2))

                poison_metrics.append(torch.stack([loss_per_sample.cpu().detach(), idx.cpu().detach()], dim=1))
                loss = loss_per_sample.mean()
                loss.backward()
                self.optimizer.step()
                
                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
                # Store batch-level training loss
                self.surrogate_train_losses.append({
                    'epoch': epoch,
                    'batch': batch_index,
                    'loss': batch_loss,
                    'phase': 'surrogate_training'
                })
            
            # Log average training loss for the epoch
            avg_loss = np.mean(epoch_losses)
            self.surrogate_epoch_train_losses.append({
                'epoch': epoch,
                'avg_loss': avg_loss,
                'phase': 'surrogate_training'
            })
            if self.experiment is not None:
                self.experiment.log_metric("surrogate_train_loss", avg_loss, step=epoch, epoch=epoch)

            # Reduce learning rate on plateau
            if self.scheduler is not None:
                self.scheduler.step(avg_loss)
            
            # Early stopping check
            will_early_stop = False
            if avg_loss < self.surrogate_best_loss:
                self.surrogate_best_loss = avg_loss
                self.surrogate_patience_counter = 0
            else:
                self.surrogate_patience_counter += 1
                if self.surrogate_patience_counter >= self.early_stopping_patience:
                    print(f'[STATUS] Early stopping triggered at epoch {epoch}: no improvement for {self.early_stopping_patience} epochs')
                    print(f'[STATUS] Best training loss: {self.surrogate_best_loss:.6f} at epoch {epoch - self.surrogate_patience_counter}')
                    will_early_stop = True

            if epoch > self.warmup:
                self.attacker.update_trigger_generator(self.net, 
                                                       epoch, 
                                                       self.num_epochs, 
                                                       use_timestamps=self.use_timestamps)

            # Only save predictions on the last epoch or when early stopping
            is_last_epoch = (epoch == self.num_epochs - 1) or will_early_stop
            self.validate(self.net, epoch, self.warmup, save_predictions=is_last_epoch)
            
            if will_early_stop:
                break

    def validate(self, model, epoch, atk_eval_epoch=0, phase='surrogate', save_predictions=False):
        model.eval()
        self.attacker.eval()
        cln_info = atk_info = ''
        with torch.no_grad():
            cln_preds = []
            atk_preds = []
            cln_targets = []
            atk_targets = []

            for batch_index, batch_data in enumerate(self.cln_test_loader):
                # calculate the clean performance
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.cln_test_set.denormalize(outputs)
                # Always compute metrics, but only store arrays if save_predictions is True
                labels_np = labels.cpu().detach().numpy()
                outputs_np = outputs.cpu().detach().numpy()
                if save_predictions:
                    cln_targets.append(labels_np)
                    cln_preds.append(outputs_np)
                else:
                    # Store temporarily just for metric computation
                    cln_targets.append(labels_np)
                    cln_preds.append(outputs_np)

            # Compute metrics from collected predictions/targets
            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_targets = np.concatenate(cln_targets, axis=0)
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_rmse = mean_squared_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1)) ** 0.5
            
            # Only keep arrays if save_predictions is True
            if not save_predictions:
                cln_preds = None
                cln_targets = None

            cln_info = f' | clean MAE: {cln_mae}, clean RMSE: {cln_rmse}'
            
            # Initialize metrics dict - only store predictions/targets if save_predictions is True
            metrics_dict = {
                'epoch': epoch,
                'phase': phase,
                'clean_mae': cln_mae,
                'clean_rmse': cln_rmse,
                'attacked_mae': None,
                'attacked_rmse': None,
                'clean_prediction': cln_preds if save_predictions else None,  # Only store if save_predictions
                'clean_target': cln_targets if save_predictions else None,   # Only store if save_predictions
                'attacked_prediction': None,    # Will be set if attacked metrics exist
                'attacked_target': None         # Will be set if attacked metrics exist
            }
            
            # Log clean metrics to Comet ML
            if self.experiment is not None:
                self.experiment.log_metric(f"{phase}_clean_mae", cln_mae, step=epoch, epoch=epoch)
                self.experiment.log_metric(f"{phase}_clean_rmse", cln_rmse, step=epoch, epoch=epoch)

            if epoch > atk_eval_epoch:
                for batch_index, batch_data in enumerate(self.atk_test_loader):
                    # calculate the attacked performance (or clean performance in no-attack mode)
                    if not self.use_timestamps:
                        encoder_inputs, labels, clean_labels, idx = batch_data
                        x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                    else:
                        encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                    labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                    x_des = torch.zeros_like(labels)
                    outputs = model(encoder_inputs, x_mark, x_des, None)
                    outputs = self.atk_test_set.denormalize(outputs)
                    
                    labels = labels[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    outputs = outputs[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    
                    # Always compute metrics, but only store arrays if save_predictions is True
                    labels_np = labels.cpu().detach().numpy()
                    outputs_np = outputs.cpu().detach().numpy()
                    atk_targets.append(labels_np)
                    atk_preds.append(outputs_np)

                # Compute metrics from collected predictions/targets
                atk_preds = np.concatenate(atk_preds, axis=0)
                atk_targets = np.concatenate(atk_targets, axis=0)
                atk_mae = mean_absolute_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1))
                atk_rmse = mean_squared_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1)) ** 0.5
                
                # Only keep arrays if save_predictions is True
                if not save_predictions:
                    atk_preds = None
                    atk_targets = None

                atk_info = f' | attacked MAE: {atk_mae}, attacked RMSE: {atk_rmse}'
                
                # Update metrics dict with attack metrics - only store arrays if save_predictions is True
                metrics_dict['attacked_mae'] = atk_mae
                metrics_dict['attacked_rmse'] = atk_rmse
                metrics_dict['attacked_prediction'] = atk_preds if save_predictions else None
                metrics_dict['attacked_target'] = atk_targets if save_predictions else None
                # Log attack metrics to Comet ML
                if self.experiment is not None:
                    self.experiment.log_metric(f"{phase}_attacked_mae", atk_mae, step=epoch, epoch=epoch)
                    self.experiment.log_metric(f"{phase}_attacked_rmse", atk_rmse, step=epoch, epoch=epoch)
                    # self.experiment.log_metric(f"{phase}_attacked_prediction", atk_preds, step=epoch, epoch=epoch)
                    # self.experiment.log_metric(f"{phase}_attacked_target", atk_targets, step=epoch, epoch=epoch)
            
            # Store test metrics
            if phase == 'surrogate':
                self.surrogate_test_metrics.append(metrics_dict)
            elif phase == 'victim':
                self.victim_test_metrics.append(metrics_dict)

        info = 'Epoch: {}'.format(epoch) + cln_info + atk_info
        print(info)

    def test(self):
        # Reset early stopping counters for new training run
        self.victim_best_loss = float('inf')
        self.victim_patience_counter = 0
        
        self.attacker.eval()
        
        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        reduce_lr_patience = getattr(self.config, 'reduce_lr_patience', None)
        if reduce_lr_patience is None and hasattr(self.config, 'Train'):
            reduce_lr_patience = getattr(self.config.Train, 'reduce_lr_patience', None)
        if reduce_lr_patience is not None:
            victim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=reduce_lr_patience
            )
        else:
            victim_scheduler = None

        self.attacker.sparse_inject()
        self.train_set = self.attacker.dataset
        self.train_loader = DataLoader(self.train_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)
        for epoch in range(self.num_epochs):
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training new forecasting model {epoch}/{self.num_epochs}')
            epoch_losses = []
            for batch_index, batch_data in enumerate(pbar):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data

                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                optimizer.zero_grad()

                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)

                loss = F.smooth_l1_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
                # Store batch-level training loss
                self.victim_train_losses.append({
                    'epoch': epoch,
                    'batch': batch_index,
                    'loss': batch_loss,
                    'phase': 'victim_training'
                })
            
            # Log average training loss for the victim model
            avg_loss = np.mean(epoch_losses)
            self.victim_epoch_train_losses.append({
                'epoch': epoch,
                'avg_loss': avg_loss,
                'phase': 'victim_training'
            })
            if self.experiment is not None:
                self.experiment.log_metric("victim_train_loss", avg_loss, step=epoch, epoch=epoch)

            # Reduce learning rate on plateau
            if victim_scheduler is not None:
                victim_scheduler.step(avg_loss)
            
            # Early stopping check
            will_early_stop = False
            if avg_loss < self.victim_best_loss:
                self.victim_best_loss = avg_loss
                self.victim_patience_counter = 0
            else:
                self.victim_patience_counter += 1
                if self.victim_patience_counter >= self.early_stopping_patience:
                    print(f'[STATUS] Early stopping triggered at epoch {epoch}: no improvement for {self.early_stopping_patience} epochs')
                    print(f'[STATUS] Best training loss: {self.victim_best_loss:.6f} at epoch {epoch - self.victim_patience_counter}')
                    will_early_stop = True

            # Only save predictions on the last epoch or when early stopping
            is_last_epoch = (epoch == self.num_epochs - 1) or will_early_stop
            self.validate(model, epoch, 0, phase='victim', save_predictions=is_last_epoch)
            
            if will_early_stop:
                break
    
    def save_results_to_dataframe(self, experiment_id, output_dir='log_results'):
        """
        Save all losses and metrics to dataframes and save them to CSV files.
        Predictions and targets are saved as separate .npy files for proper serialization.
        
        Args:
            experiment_id: Unique identifier for this experiment
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        # Create subdirectory for numpy arrays
        arrays_dir = os.path.join(output_dir, f'{experiment_id}_arrays')
        os.makedirs(arrays_dir, exist_ok=True)
        
        # Combine all training losses (batch-level)
        all_train_losses = self.surrogate_train_losses + self.victim_train_losses
        if all_train_losses:
            train_df = pd.DataFrame(all_train_losses)
            train_path = os.path.join(output_dir, f'{experiment_id}_train_losses.csv')
            train_df.to_csv(train_path, index=False)
            print(f'[INFO] Saved training losses to {train_path}')
        
        # Combine all epoch-level training losses
        all_epoch_train_losses = self.surrogate_epoch_train_losses + self.victim_epoch_train_losses
        if all_epoch_train_losses:
            epoch_train_df = pd.DataFrame(all_epoch_train_losses)
            epoch_train_path = os.path.join(output_dir, f'{experiment_id}_epoch_train_losses.csv')
            epoch_train_df.to_csv(epoch_train_path, index=False)
            print(f'[INFO] Saved epoch training losses to {epoch_train_path}')
        
        # Process test metrics: save arrays as .npy files and store paths in CSV
        all_test_metrics = self.surrogate_test_metrics + self.victim_test_metrics
        if all_test_metrics:
            # Create a copy for CSV (with file paths instead of arrays)
            test_metrics_for_csv = []
            for idx, item in enumerate(all_test_metrics):
                csv_item = item.copy()
                
                # Save clean predictions and targets
                if item.get('clean_prediction') is not None and isinstance(item['clean_prediction'], np.ndarray):
                    clean_pred_path = os.path.join(arrays_dir, f'clean_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    np.save(clean_pred_path, item['clean_prediction'])
                    csv_item['clean_prediction_path'] = clean_pred_path
                    csv_item['clean_prediction'] = None  # Remove array from CSV data
                else:
                    csv_item['clean_prediction_path'] = None
                    # Backward compatibility: check old 'prediction' key
                    if item.get('prediction') is not None and isinstance(item['prediction'], np.ndarray):
                        clean_pred_path = os.path.join(arrays_dir, f'clean_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                        np.save(clean_pred_path, item['prediction'])
                        csv_item['clean_prediction_path'] = clean_pred_path
                
                if item.get('clean_target') is not None and isinstance(item['clean_target'], np.ndarray):
                    clean_target_path = os.path.join(arrays_dir, f'clean_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    np.save(clean_target_path, item['clean_target'])
                    csv_item['clean_target_path'] = clean_target_path
                    csv_item['clean_target'] = None  # Remove array from CSV data
                else:
                    csv_item['clean_target_path'] = None
                    # Backward compatibility: check old 'target' or 'labels' keys
                    target_key = None
                    if 'target' in item:
                        target_key = 'target'
                    elif 'labels' in item:
                        target_key = 'labels'
                    if target_key and item.get(target_key) is not None and isinstance(item[target_key], np.ndarray):
                        clean_target_path = os.path.join(arrays_dir, f'clean_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                        np.save(clean_target_path, item[target_key])
                        csv_item['clean_target_path'] = clean_target_path
                
                # Save attacked predictions and targets
                if item.get('attacked_prediction') is not None and isinstance(item['attacked_prediction'], np.ndarray):
                    atk_pred_path = os.path.join(arrays_dir, f'attacked_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    np.save(atk_pred_path, item['attacked_prediction'])
                    csv_item['attacked_prediction_path'] = atk_pred_path
                    csv_item['attacked_prediction'] = None  # Remove array from CSV data
                else:
                    csv_item['attacked_prediction_path'] = None
                
                if item.get('attacked_target') is not None and isinstance(item['attacked_target'], np.ndarray):
                    atk_target_path = os.path.join(arrays_dir, f'attacked_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    np.save(atk_target_path, item['attacked_target'])
                    csv_item['attacked_target_path'] = atk_target_path
                    csv_item['attacked_target'] = None  # Remove array from CSV data
                else:
                    csv_item['attacked_target_path'] = None
                
                # Clean up old keys for backward compatibility
                for old_key in ['prediction', 'target', 'labels']:
                    if old_key in csv_item and isinstance(csv_item[old_key], np.ndarray):
                        csv_item[old_key] = None
                
                test_metrics_for_csv.append(csv_item)
            
            test_df = pd.DataFrame(test_metrics_for_csv)
            test_path = os.path.join(output_dir, f'{experiment_id}_test_metrics.csv')
            test_df.to_csv(test_path, index=False)
            print(f'[INFO] Saved test metrics to {test_path}')
            print(f'[INFO] Saved prediction/target arrays to {arrays_dir}')
        
        # Create a comprehensive dataframe with all results
        all_results = []
        
        # Add training losses
        for item in all_train_losses:
            all_results.append({
                'epoch': item['epoch'],
                'batch': item.get('batch', None),
                'prediction_path': None,
                'target_path': None,
                'metric_type': 'train_loss',
                'metric_value': item['loss'],
                'phase': item['phase']
            })
        
        # Add epoch-level training losses
        for item in all_epoch_train_losses:
            all_results.append({
                'epoch': item['epoch'],
                'batch': None,
                'prediction_path': None,
                'target_path': None,
                'metric_type': 'epoch_avg_train_loss',
                'metric_value': item['avg_loss'],
                'phase': item['phase']
            })
        
        # Add test metrics
        for idx, item in enumerate(all_test_metrics):
            # Handle clean metrics
            if item['clean_mae'] is not None:
                clean_pred_path = None
                clean_target_path = None
                # Use new keys if available, otherwise fall back to old keys for backward compatibility
                if item.get('clean_prediction') is not None and isinstance(item['clean_prediction'], np.ndarray):
                    clean_pred_path = os.path.join(arrays_dir, f'clean_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    if not os.path.exists(clean_pred_path):  # Avoid duplicate saves
                        np.save(clean_pred_path, item['clean_prediction'])
                elif item.get('prediction') is not None and isinstance(item['prediction'], np.ndarray):
                    clean_pred_path = os.path.join(arrays_dir, f'clean_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    if not os.path.exists(clean_pred_path):
                        np.save(clean_pred_path, item['prediction'])
                
                if item.get('clean_target') is not None and isinstance(item['clean_target'], np.ndarray):
                    clean_target_path = os.path.join(arrays_dir, f'clean_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    if not os.path.exists(clean_target_path):
                        np.save(clean_target_path, item['clean_target'])
                else:
                    # Backward compatibility
                    target_key = 'target' if 'target' in item else ('labels' if 'labels' in item else None)
                    if target_key and item.get(target_key) is not None and isinstance(item[target_key], np.ndarray):
                        clean_target_path = os.path.join(arrays_dir, f'clean_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                        if not os.path.exists(clean_target_path):
                            np.save(clean_target_path, item[target_key])
                
                all_results.append({
                    'epoch': item['epoch'],
                    'batch': None,
                    'prediction_path': clean_pred_path,
                    'target_path': clean_target_path,
                    'metric_type': 'clean_mae',
                    'metric_value': item['clean_mae'],
                    'phase': item['phase']
                })
                all_results.append({
                    'epoch': item['epoch'],
                    'batch': None,
                    'prediction_path': clean_pred_path,
                    'target_path': clean_target_path,
                    'metric_type': 'clean_rmse',
                    'metric_value': item['clean_rmse'],
                    'phase': item['phase']
                })
            
            # Handle attacked metrics
            if item['attacked_mae'] is not None:
                atk_pred_path = None
                atk_target_path = None
                if item.get('attacked_prediction') is not None and isinstance(item['attacked_prediction'], np.ndarray):
                    atk_pred_path = os.path.join(arrays_dir, f'attacked_predictions_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    if not os.path.exists(atk_pred_path):  # Avoid duplicate saves
                        np.save(atk_pred_path, item['attacked_prediction'])
                
                if item.get('attacked_target') is not None and isinstance(item['attacked_target'], np.ndarray):
                    atk_target_path = os.path.join(arrays_dir, f'attacked_targets_epoch{item["epoch"]}_{item["phase"]}_{idx}.npy')
                    if not os.path.exists(atk_target_path):
                        np.save(atk_target_path, item['attacked_target'])
                
                all_results.append({
                    'epoch': item['epoch'],
                    'batch': None,
                    'prediction_path': atk_pred_path,
                    'target_path': atk_target_path,
                    'metric_type': 'attacked_mae',
                    'metric_value': item['attacked_mae'],
                    'phase': item['phase']
                })
                all_results.append({
                    'epoch': item['epoch'],
                    'batch': None,
                    'prediction_path': atk_pred_path,
                    'target_path': atk_target_path,
                    'metric_type': 'attacked_rmse',
                    'metric_value': item['attacked_rmse'],
                    'phase': item['phase']
                })
        
        if all_results:
            comprehensive_df = pd.DataFrame(all_results)
            comprehensive_path = os.path.join(output_dir, f'{experiment_id}_all_results.csv')
            comprehensive_df.to_csv(comprehensive_path, index=False)
            print(f'[INFO] Saved comprehensive results to {comprehensive_path}')
        
        return comprehensive_df if all_results else None
    
    @staticmethod
    def load_predictions_and_targets(csv_path, experiment_id=None, output_dir='log_results'):
        """
        Load predictions and targets from saved CSV and .npy files.
        
        Args:
            csv_path: Path to the CSV file containing metrics
            experiment_id: Optional experiment ID (if None, inferred from csv_path)
            output_dir: Directory where arrays are stored (default: 'log_results')
        
        Returns:
            DataFrame with loaded predictions and targets as numpy arrays
            
        Example:
            # After saving results with trainer.save_results_to_dataframe('exp1')
            df = Trainer.load_predictions_and_targets('log_results/exp1_test_metrics.csv')
            # Access predictions and targets as numpy arrays
            for idx, row in df.iterrows():
                if row['prediction'] is not None:
                    pred = row['prediction']  # numpy array
                    target = row['target']    # numpy array
        """
        df = pd.read_csv(csv_path)
        
        # Infer experiment_id and arrays directory if not provided
        if experiment_id is None:
            # Try to infer from csv_path
            basename = os.path.basename(csv_path)
            if '_test_metrics.csv' in basename:
                experiment_id = basename.replace('_test_metrics.csv', '')
            elif '_all_results.csv' in basename:
                experiment_id = basename.replace('_all_results.csv', '')
            else:
                # Default: use parent directory name
                experiment_id = os.path.splitext(basename)[0]
        
        arrays_dir = os.path.join(output_dir, f'{experiment_id}_arrays')
        
        # Helper function to load arrays from paths
        def load_array_from_path(path):
            if pd.isna(path) or path is None:
                return None
            if isinstance(path, str) and os.path.exists(path):
                try:
                    return np.load(path)
                except:
                    return None
            return None
        
        # Load clean predictions and targets
        if 'clean_prediction_path' in df.columns:
            df['clean_prediction'] = df['clean_prediction_path'].apply(load_array_from_path)
        if 'clean_target_path' in df.columns:
            df['clean_target'] = df['clean_target_path'].apply(load_array_from_path)
        
        # Load attacked predictions and targets
        if 'attacked_prediction_path' in df.columns:
            df['attacked_prediction'] = df['attacked_prediction_path'].apply(load_array_from_path)
        if 'attacked_target_path' in df.columns:
            df['attacked_target'] = df['attacked_target_path'].apply(load_array_from_path)
        
        # Backward compatibility: handle old format with single prediction/target paths
        if 'prediction_path' in df.columns and 'clean_prediction' not in df.columns:
            df['prediction'] = df['prediction_path'].apply(load_array_from_path)
            # Try to infer if it's clean or attacked based on metric_type
            if 'metric_type' in df.columns:
                df['clean_prediction'] = df.apply(
                    lambda row: row['prediction'] if pd.notna(row['metric_type']) and 'clean' in str(row['metric_type']) else None,
                    axis=1
                )
                df['attacked_prediction'] = df.apply(
                    lambda row: row['prediction'] if pd.notna(row['metric_type']) and 'attacked' in str(row['metric_type']) else None,
                    axis=1
                )
        
        if 'target_path' in df.columns and 'clean_target' not in df.columns:
            df['target'] = df['target_path'].apply(load_array_from_path)
            # Try to infer if it's clean or attacked based on metric_type
            if 'metric_type' in df.columns:
                df['clean_target'] = df.apply(
                    lambda row: row['target'] if pd.notna(row['metric_type']) and 'clean' in str(row['metric_type']) else None,
                    axis=1
                )
                df['attacked_target'] = df.apply(
                    lambda row: row['target'] if pd.notna(row['metric_type']) and 'attacked' in str(row['metric_type']) else None,
                    axis=1
                )
        
        # Backward compatibility: try to load from old string format
        for col in ['prediction', 'target', 'labels']:
            if col in df.columns and df[col].dtype == 'object':
                def try_load_old_format(val):
                    if pd.isna(val) or val is None:
                        return None
                    if isinstance(val, np.ndarray):
                        return val
                    if isinstance(val, str):
                        try:
                            # Try to load from path if it's a path string
                            if os.path.exists(val):
                                return np.load(val)
                            # Otherwise try to parse as array string
                            from ast import literal_eval
                            return np.array(literal_eval(val))
                        except:
                            return None
                    return None
                df[col] = df[col].apply(try_load_old_format)
        
        # Clean up duplicate columns
        if 'labels' in df.columns and 'target' in df.columns:
            # Keep target, remove labels if they're the same
            df = df.drop(columns=['labels'])
        
        return df