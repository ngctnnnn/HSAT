import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import optim
import numpy as np
from math import ceil
import tqdm
from trigger import TgrGCN, InverseTgr
import random

from torch.utils.data import DataLoader


def fft_compress(raw_data_seq, n_components=200):
    """
    compress the time series data using fft to have global representation for each variable.
    """
    if len(raw_data_seq.shape) == 2:
        raw_data_seq = raw_data_seq[:, :, None]
    data_seq = raw_data_seq[:, :, 0:1]
    # data_seq: (l, n, c)
    l, n, c = data_seq.shape
    data_seq = data_seq.reshape(l, -1).transpose()
    # use fft to have the amplitude, phase, and frequency for each time series data
    fft_data = np.fft.fft(data_seq, axis=1)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    frequency = np.fft.fftfreq(l)

    # choose the top n_components frequency components
    top_indices = np.argsort(amplitude, axis=1)[::-1][:, :n_components]
    amplitude_top = amplitude[np.arange(amplitude.shape[0])[:, None], top_indices]
    phase_top = phase[np.arange(phase.shape[0])[:, None], top_indices]
    frequency_top = frequency[top_indices]
    feature_top = np.concatenate([amplitude_top, phase_top, frequency_top], axis=1)
    return feature_top


class Attacker:
    def __init__(self, dataset, channel_features, atk_vars, atk_vars_std, atk_vars_mean, config, target_pattern,
                 train_std, train_mean,batch_size=512,device='cuda',
                 trigger_name='TgrGCN'):
        """
        the attacker class is used to inject triggers and target patterns into the dataset.
        the attacker class have the full access to the dataset and the trigger generator.
        """
        self.device = device
        self.dataset = dataset
        self.train_std = train_std
        self.train_mean=train_mean
        self.batch_size=batch_size

        self.atk_vars_std = atk_vars_std
        self.atk_vars_mean = atk_vars_mean

        self.target_pattern = target_pattern
        self.atk_vars = atk_vars
        self.trigger_name = trigger_name

        

        if self.trigger_name == 'TgrGCN':
            self.trigger_generator = TgrGCN(config, sim_feats=channel_features, atk_vars=atk_vars, device=device)
        elif self.trigger_name == 'InverseTgr':
            self.trigger_generator = InverseTgr(config, input_dim=12, atk_vars=atk_vars, device=device)

        self.trigger_len = config.trigger_len
        self.pattern_len = config.pattern_len
        self.bef_tgr_len = config.bef_tgr_len  # the length of the data before the trigger to generate the trigger

        self.fct_input_len = config.Dataset.len_input  # the length of the input for the forecast model
        self.fct_output_len = config.Dataset.num_for_predict  # the length of the output for the forecast model
        self.alpha_t = config.alpha_t
        self.alpha_s = config.alpha_s
        self.temporal_poison_num = ceil(self.alpha_t * len(self.dataset))

        self.trigger_generator = self.trigger_generator.to(device)
        self.attack_optim = optim.Adam(self.trigger_generator.parameters(), lr=config.attack_lr)
        self.atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.attack_optim, milestones=[20, 40], gamma=0.9)

      
        self.lam_norm =0.1 #config.lam_norm
        self.lambda_offset = torch.tensor(0.05).to(device)
        self.lambda_clean = 0.1
        self.lambda_attack = 1


      

        

    def state_dict(self):
        attacker_state = {
            'target_pattern': self.target_pattern.cpu().detach().numpy(),
            'trigger_generator': self.trigger_generator.state_dict(),
            'trigger_len': self.trigger_len,
            'pattern_len': self.pattern_len,
            'bef_tgr_len': self.bef_tgr_len,
            'fct_input_len': self.fct_input_len,
            'fct_output_len': self.fct_output_len,
            'alpha_t': self.alpha_t,
            'alpha_s': self.alpha_s,
            'temporal_poison_num': self.temporal_poison_num,
            
            'attack_optim': self.attack_optim.state_dict(),
            'atk_scheduler': self.atk_scheduler.state_dict(),
            # Fix: save list directly (no need to convert to tensor)
            'delta_t_lists': self.delta_t_lists if hasattr(self, 'delta_t_lists') else None,
            
            'atk_ts': self.atk_ts.cpu().detach().numpy() if hasattr(self, 'atk_ts') else None,
            'atk_vars': self.atk_vars.cpu().detach().numpy() if hasattr(self, 'atk_vars') else None,

            'trigger_name': self.trigger_name,  # trigger name

        }
        return attacker_state

    def load_state_dict(self, attacker_state):
        self.trigger_len = attacker_state['trigger_len']
        self.pattern_len = attacker_state['pattern_len']
        self.bef_tgr_len = attacker_state['bef_tgr_len']
        self.fct_input_len = attacker_state['fct_input_len']
        self.fct_output_len = attacker_state['fct_output_len']
        self.alpha_t = attacker_state['alpha_t']
        self.alpha_s = attacker_state['alpha_s']
        self.temporal_poison_num = attacker_state['temporal_poison_num']
        

        # Trigger name
        self.trigger_name = attacker_state['trigger_name']

        self.trigger_generator.load_state_dict(attacker_state['trigger_generator'])
        self.attack_optim.load_state_dict(attacker_state['attack_optim'])
        self.atk_scheduler.load_state_dict(attacker_state['atk_scheduler'])
        self.target_pattern = torch.from_numpy(attacker_state['target_pattern'])
        self.atk_ts = torch.from_numpy(attacker_state['atk_ts']) if attacker_state['atk_ts'] is not None else None
        self.atk_vars = torch.from_numpy(attacker_state['atk_vars']) if attacker_state['atk_vars'] is not None else None

        self.trigger_generator = self.trigger_generator.to(self.device)
        self.target_pattern = self.target_pattern.to(self.device)

        # Load delta_t_lists (native list, no need to convert to tensor)
        if attacker_state['delta_t_lists'] is not None:
            self.delta_t_lists = attacker_state['delta_t_lists']  # assign list directly
        else:
            self.delta_t_lists = []  # default to empty list

        if self.atk_ts is not None:
            self.atk_ts = self.atk_ts.to(self.device)
        if self.atk_vars is not None:
            self.atk_vars = self.atk_vars.to(self.device)

    def eval(self):
        self.trigger_generator.eval()

    def train(self):
        self.trigger_generator.train()

    def train_inverse_tgr(self, train_data, use_timestamps, EPOCH, BATCH_SIZE):
        self.train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)

        # Training loop
        for epoch in range(EPOCH):
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training data {epoch}/{EPOCH}')
            for batch_index, batch_data in enumerate(pbar):
                if not use_timestamps:
                    # already normalized
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                # torch.squeeze(encoder_inputs) shape is batch*f*time; .permute(0, 2, 1) gives batch*time*f
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                self.attack_optim.zero_grad()

                x_des = torch.zeros_like(labels)
                outputs = self.trigger_generator(encoder_inputs, x_mark, x_des, None)
                outputs = train_data.denormalize(outputs)

                labels=labels[:,-self.trigger_len:, :]  # keep only labels for prediction length
                # flip along time dimension
                labels = labels.flip(dims=[1])  # reverse time dimension

                loss = F.smooth_l1_loss(outputs, labels, reduction='mean')
                # loss_per_sample = loss_per_sample.mean(dim=(1, 2))
                # loss = loss_per_sample.mean()
                loss.backward()

                self.attack_optim.step()
                pbar.set_postfix({'loss': loss.item()})


    
    def generate_delta_t_lists(self):
        
        assert hasattr(self, 'atk_vars'), 'Please set the attack variable first.'
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'

        n = len(self.atk_vars)
        pattern_len = self.target_pattern.shape[-1]

        self.delta_t_lists = []

         
        for _ in self.atk_ts.tolist():
             
            delta_t_list = [random.randint(0, self.fct_output_len - pattern_len) for _ in range(n)]
            self.delta_t_lists.append(delta_t_list)

    def set_atk_timestamp(self, atk_ts):
        """
        set the attack timestamp for the attacker.
        """
        self.atk_ts = atk_ts

        # generate delta_t_list
        self.generate_delta_t_lists()

    def set_atk_variables(self, atk_var):
        self.atk_vars = atk_var

    def set_atk(self, atk_ts, atk_var):
        self.set_atk_timestamp(atk_ts)
        self.set_atk_variables(atk_var)

   
    
    def sparse_inject_batch(self):
        assert hasattr(self, 'atk_vars'), 'Please set the attack variable first.'
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'
        assert hasattr(self, 'delta_t_lists'), 'Please generate delta_t_lists first.'

        self.dataset.init_poison_data()

        if self.dataset.use_timestamp:
            slices, slice_timestamps, delta_t_lists = self.get_trigger_slices(
                bef_len=self.fct_input_len - self.trigger_len,
                aft_len=self.trigger_len + self.fct_output_len
            )
        else:
            slices, delta_t_lists = self.get_trigger_slices(
                bef_len=self.fct_input_len - self.trigger_len,
                aft_len=self.trigger_len + self.fct_output_len
            )
            slice_timestamps = None

        total = len(slices)
        batch_size=self.batch_size
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)

            #print(total, batch_start, batch_end)
            batch_slice_ids = list(range(batch_start, batch_end))

            batch_slices_list = slices[batch_start:batch_end]
            batch_delta_t_lists = delta_t_lists[batch_start:batch_end]

            # [feature, batch, time]
            batch_slices = torch.stack(batch_slices_list, dim=1)[:,:,0,:]

            # insert target pattern (per sample, per variable)
            for i, idx in enumerate(batch_slice_ids):
                ts = self.atk_ts[idx].item()
                delta_t_list = delta_t_lists[idx]
                for j, var in enumerate(self.atk_vars):
                    delta_t = delta_t_list[j]
                    s = ts + self.trigger_len + delta_t
                    e = s + self.pattern_len
                    if e <= self.dataset.poisoned_data.shape[-1]:
                        base = self.dataset.poisoned_data[var, 0, s - 1]  # previous value
                        self.dataset.poisoned_data[var, 0, s:e] = self.target_pattern[j].to(self.device) + base

            # generate triggers and perturbations (batch-supported)
            triggers, _ = self.generate_triggers_and_perturbations(
                batch_slices,
                batch_delta_t_lists, 
                batch_slice_ids,
                use_timestamps=self.dataset.use_timestamp,
                slice_timestamps=slice_timestamps
            )

            

            # insert triggers (each variable's trigger)
            for i, idx in enumerate(batch_slice_ids):
                ts = self.atk_ts[idx].item()
                self.dataset.poisoned_data[self.atk_vars, 0:1, ts:ts + self.trigger_len] = \
                    triggers[:, i:i + 1, :].detach()
               
                
            

    def predict_trigger(self, data_bef_trigger,delta_t_encoding=None):
        """
        predict the trigger using the trigger generator.
        n = number of samples, c = number of variables, l = length of the data
        :param data_bef_trigger: the data before the trigger, shape: (n, c, l).
        :return: the predicted trigger, shape: (n, c, trigger_len)
        """
        c, l = data_bef_trigger.shape[-2:]
        data_bef_trigger = self.dataset.normalize(data_bef_trigger)
        data_bef_trigger = data_bef_trigger.view(-1, self.trigger_generator.input_dim)
        triggers, perturbations = self.trigger_generator(data_bef_trigger,delta_t_embd=delta_t_encoding)
        triggers = self.dataset.denormalize(triggers).reshape(-1, c, self.trigger_len)
        return triggers, perturbations


    def get_trigger_slices(self, bef_len, aft_len,s='s1'):
        """
        A easy implementation to limit the range for soft identification.
        find all the sliced time window that contains the trigger.
        :return: a list of slices, corresponding timestamps (if used), and corresponding delta_t lists
        """
        slices = []
        timestamps = []
        delta_t_lists = []  # store delta_t_list corresponding to each slice

        # ensure delta_t_lists has been generated
        assert hasattr(self, 'delta_t_lists') and len(self.delta_t_lists) == len(self.atk_ts), \
            'Please call generate_delta_t_lists first to generate delta_t lists.'
        if s=='s1':
            for idx, ts in enumerate(self.atk_ts.tolist()):
                if ts + aft_len < self.dataset.poisoned_data.shape[-1] and ts - bef_len >= 0:
                    slices.append(self.dataset.poisoned_data[..., ts - bef_len:ts + aft_len].detach())
                    delta_t_lists.append(self.delta_t_lists[idx])  # append corresponding delta_t_list

                    if self.dataset.use_timestamp:
                        timestamps.append(self.dataset.timestamps[ts - bef_len:ts + aft_len])  # full timestamp range
        elif s=='s2':
            for idx, ts in enumerate(self.atk_ts.tolist()):
                if ts + aft_len < self.dataset.poisoned_data.shape[-1] and ts - bef_len >= 0:

                    for k in range(0,self.fct_output_len-self.pattern_len+1):

                        slices.append(self.dataset.poisoned_data[..., ts - bef_len:ts + aft_len].detach())
                        delta_t_lists.append([k for _ in range(len(self.atk_vars))])  # append corresponding delta_t_list

                        if self.dataset.use_timestamp:
                            timestamps.append(self.dataset.timestamps[ts - bef_len:ts + aft_len])  # full timestamp range

        if not self.dataset.use_timestamp:
            return slices, delta_t_lists
        return slices, timestamps, delta_t_lists
    

    def select_atk_timestamp(self, poison_metrics):
        """
        select the attack timestamp using the poison metrics (clean MAE). poison_metrics: a list of [mae, idx]
        """
        select_pos_mark = torch.zeros(len(self.dataset), dtype=torch.int)
        poison_metrics = torch.cat(poison_metrics, dim=0).to(self.device)

        sort_idx = torch.argsort(poison_metrics[:, 0], descending=True).detach().cpu().numpy()
        # ensure the distance between two poison indices is larger than trigger length + pattern length, avoid overlap
        valid_idx = []
        for i in range(len(sort_idx)):
            # use greedy algorithm to select the valid indices with the largest poison metrics
            beg_idx = int(poison_metrics[sort_idx[i], 1])
            end_idx = beg_idx + self.trigger_len + self.pattern_len + 8  # 8: the magic number to avoid overlap
            if torch.sum(select_pos_mark[beg_idx:end_idx]) == 0 and \
                    end_idx < len(self.dataset) and beg_idx > self.bef_tgr_len:
                valid_idx.append(sort_idx[i])
                select_pos_mark[beg_idx:end_idx] = 1
            if len(valid_idx) > 2 * self.temporal_poison_num:
                print('break due to enough valid indices')
                break

        valid_idx = np.array(valid_idx)
        # random select the temporal poison indices. add randomness to avoid overfitting
        top_sort_idx = np.random.choice(valid_idx, min(self.temporal_poison_num, valid_idx.shape[0]), replace=False)
        top_sort_idx = torch.from_numpy(top_sort_idx).to(self.device)
        atk_ts = poison_metrics[top_sort_idx, 1].long()
        # sort poison indices
        atk_ts = torch.sort(atk_ts)[0]
        # atk_ts: tensor of size 948, range [100, 28000], dtype torch.int64, device cuda:0

        # atk_ts=torch.randint(low=100, high=28000, size=(948,), dtype=torch.int64, device='cuda:0')

        self.set_atk_timestamp(atk_ts)

    def generate_triggers_and_perturbations(self, slice, delta_t_list, slice_id, use_timestamps, slice_timestamps=None,
                                            target_stamps=None):
        """Generate triggers and perturbations with batch input support."""
        batch_size = slice.shape[1]
        num_atk_vars = len(self.atk_vars)
        device = self.device

        if self.trigger_name == 'TgrGCN':
            # Gaussian encoding
            delta_range = self.fct_output_len - self.pattern_len + 1
            sigma = 1.0
            delta_t_encoding = torch.zeros(batch_size, num_atk_vars, delta_range, device=device)

            for b in range(batch_size):
                for j, delta_t in enumerate(delta_t_list[b]):
                    pos = torch.arange(delta_range, device=device).float()
                    center = torch.tensor(delta_t, device=device).float()
                    gaussian = torch.exp(- (pos - center) ** 2 / (2 * sigma ** 2))
                    delta_t_encoding[b, j, :] = gaussian / gaussian.sum()

            # ---- 1. compute normalized scale alpha_j ----
            alpha = self.atk_vars_mean.clone().detach()  # shape: (num_atk_vars,)
            alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)  # normalize to [0, 1]

            # ---- 2. reshape to broadcast shape: (1, num_atk_vars, 1) ----
            alpha = alpha.view(1, -1, 1).to(delta_t_encoding.device)

            # ---- 3. weight by variable dimension (channel-wise) ----
            delta_t_encoding = delta_t_encoding * alpha  # shape unchanged, values weighted


            delta_t_encoding = delta_t_encoding.view(batch_size * num_atk_vars, -1)

           

            

            data_bef = slice[self.atk_vars, :, self.fct_input_len - self.trigger_len - self.bef_tgr_len : self.fct_input_len - self.trigger_len]
            data_bef=data_bef.reshape(-1, self.bef_tgr_len)  # [B, V, L]
            data_bef= self.dataset.normalize(data_bef)  # [B, V, L]
            #data_bef = data_bef.permute(1, 0, 2)  # [B, V, L]
            #data_bef = data_bef.reshape(batch_size * num_atk_vars, self.bef_tgr_len)

            triggers, perturbations = self.trigger_generator(data_bef, delta_t_encoding=delta_t_encoding)
            triggers = self.dataset.denormalize(triggers)
            triggers=triggers.reshape(num_atk_vars, batch_size, -1)
          

        elif self.trigger_name == 'InverseTgr':
            # Gaussian encoding
            delta_t_encoding = torch.zeros(batch_size, self.fct_output_len,num_atk_vars,  device=device)
            for b in range(batch_size):
                for j, delta_t in enumerate(delta_t_list[b]):
                    start = delta_t
                    end = min(delta_t + self.pattern_len, self.fct_output_len)
                    delta_t_encoding[b, start:end, j] = 1.0  # end index is exclusive


            data_4_fct_input = slice[self.atk_vars, :, self.fct_input_len : self.fct_input_len + self.fct_output_len].clone()
            data_4_fct_input = data_4_fct_input.permute(1, 2, 0).flip(dims=[1])
            data_4_fct_input = self.dataset.normalize(data_4_fct_input)

            if use_timestamps:
                if target_stamps is None:  # timestamps for future sequence
                    #tgr_timestamps
                    batch_x_mark = torch.stack([slice_timestamps[i][self.fct_input_len : self.fct_input_len + self.fct_output_len] for i in slice_id])
                    batch_x_mark = batch_x_mark.to(device)
                else:
                    batch_x_mark = target_stamps
            else:
                batch_x_mark = torch.zeros(batch_size, self.fct_output_len, 4, device=device)
            inverse_outputs = self.trigger_generator(
                    data_4_fct_input, batch_x_mark.flip(dims=[1]), None, None, delta_t_encoding
                )
            inverse_outputs = torch.clamp(inverse_outputs, min=-1, max=1)
            triggers = self.dataset.denormalize(inverse_outputs.flip(dims=[1])).permute(2, 0, 1)  # flip result (correct order)
            perturbations = triggers - slice[self.atk_vars, :, self.fct_input_len - self.trigger_len - 1 : self.fct_input_len - self.trigger_len]

        else:
            raise NotImplementedError(f"Unknown trigger_name: {self.trigger_name}")





        return triggers, perturbations
    

    def update_trigger_generator_refactored(self, net, use_timestamps=False,epochId=10):

        self.dataset.init_poison_data()

        if not use_timestamps:
            tgr_slices, delta_t_lists = self.get_trigger_slices(
                self.fct_input_len - self.trigger_len,
                self.trigger_len + self.fct_output_len+ self.pattern_len
            )
            slice_timestamps = None
        else:
            tgr_slices, slice_timestamps, delta_t_lists = self.get_trigger_slices(
                self.fct_input_len - self.trigger_len,
                self.trigger_len + self.fct_output_len+ + self.pattern_len
            )

       
        # pbar = tqdm.tqdm(tgr_slices, desc=f'Attacking data {epoch}/{epochs}')
        for slice_id, slice in enumerate(tgr_slices):
            slice = slice.to(self.device)[:, 0:1, :]  # adjust dims: var*batch*time
            delta_t_list = delta_t_lists[slice_id]


         
            for i, var in enumerate(self.atk_vars):
                        delta_t = delta_t_list[i]
                        # insert attack pattern into slice
                        base= slice[var, :, self.fct_input_len + delta_t-1].unsqueeze(-1)  # previous value
                        slice[var, :, self.fct_input_len + delta_t:self.fct_input_len + delta_t + self.pattern_len] = \
                            self.target_pattern[i].unsqueeze(0) + base

            # generate triggers and perturbations
            triggers, perturbations = self.generate_triggers_and_perturbations(
                slice, [delta_t_list], [slice_id], use_timestamps, slice_timestamps=slice_timestamps
            )
            slice[self.atk_vars, :, self.fct_input_len - self.trigger_len:self.fct_input_len] = triggers

           

            # compute beta weights
            beta_weights = self.compute_beta_weights(delta_t_list)

            # prepare model inputs and labels (sliding window)
            batch_inputs_bkd = [slice[..., i:i + self.fct_input_len] for i in range(self.pattern_len)]
            batch_labels_bkd = [
                slice[..., i + self.fct_input_len: i + self.fct_input_len + self.fct_output_len].detach()
                for i in range(self.pattern_len)
            ]
            batch_inputs_bkd = torch.stack(batch_inputs_bkd, dim=0)
            batch_labels_bkd = torch.stack(batch_labels_bkd, dim=0)

            # adjust input/output shapes
            batch_inputs_bkd = batch_inputs_bkd[:, :, 0:1, :]
            batch_labels_bkd = batch_labels_bkd[:, :, 0, :]
            batch_inputs_bkd = self.dataset.normalize(batch_inputs_bkd)

            
            batch_inputs_bkd = batch_inputs_bkd.squeeze(2).permute(0, 2, 1)
            batch_labels_bkd = batch_labels_bkd.permute(0, 2, 1)

            # handle timestamps
            if use_timestamps:
                batch_x_mark = [
                    slice_timestamps[slice_id][i:i + self.fct_input_len]
                    for i in range(self.pattern_len)
                ]
                batch_y_mark = [
                    slice_timestamps[slice_id][i + self.fct_input_len:
                                             i + self.fct_input_len + self.fct_output_len]
                    for i in range(self.pattern_len)
                ]
                batch_x_mark = torch.stack(batch_x_mark, dim=0)
                batch_y_mark = torch.stack(batch_y_mark, dim=0)
            else:
                batch_x_mark = torch.zeros(
                    batch_inputs_bkd.shape[0], batch_inputs_bkd.shape[1], 4
                ).to(self.device)

            # model prediction / forward pass
            self.attack_optim.zero_grad()
            x_des = torch.zeros_like(batch_labels_bkd)
            outputs_bkd = net(batch_inputs_bkd, batch_x_mark, x_des, None)
            outputs_bkd = self.dataset.denormalize(outputs_bkd)

            # compute loss and backward pass
            loss, ret_str = self.compute_loss(outputs_bkd, batch_labels_bkd, beta_weights, triggers, perturbations,delta_t_list)
            loss.backward()
            self.attack_optim.step()

        # scheduler step
        self.atk_scheduler.step()

    def build_time_masks(self, delta_t_list, i):
        """
        Dynamically generate the time mask for the i-th sliding window.
        :param delta_t_list: offset time for each attack variable
        :param i: current sliding window index
        """
        var_time_masks = {}
        current_output_start = i
        current_output_end = i + self.fct_output_len

        for j, var in enumerate(self.atk_vars):
            delta_t = delta_t_list[j]
            pattern_start = delta_t
            pattern_end = delta_t + self.pattern_len

            overlap_start = max(pattern_start, current_output_start)
            overlap_end = min(pattern_end, current_output_end)

            time_mask = torch.zeros(1, self.fct_output_len, device=self.device)

            if overlap_start < overlap_end:
                # map global mask range back to relative indices within local window
                local_start = overlap_start - current_output_start
                local_end = overlap_end - current_output_start
                time_mask[0, local_start:local_end] = 1.0

            var_time_masks[var.item()] = time_mask

        return var_time_masks

    def compute_beta_weights(self, delta_t_list):
        """Compute beta weights reflecting attack contribution per time position; offset_penalty normalized across channels."""
        beta_weights = torch.zeros((self.pattern_len, len(self.atk_vars)), device=self.device)

        # compute raw offset_penalty for all channels first
        delta_t_tensor = torch.tensor(delta_t_list, dtype=torch.float32, device=self.device)
        offset_penalties = torch.exp(self.lambda_offset * delta_t_tensor)  # shape: [num_vars]
        offset_penalties = offset_penalties / (offset_penalties.sum() + 1e-8)  # normalize across channels

        for i in range(self.pattern_len):
            for j, delta_t in enumerate(delta_t_list):
                # prediction window range (relative to input end)
                pred_start = i
                pred_end = i + self.fct_output_len

                # attack pattern range
                pattern_start = delta_t
                pattern_end = delta_t + self.pattern_len

                # compute overlap
                overlap_start = max(pred_start, pattern_start)
                overlap_end = min(pred_end, pattern_end)
                overlap_len = max(0, overlap_end - overlap_start)

                coverage_ratio = overlap_len / self.pattern_len

                # use normalized offset_penalty
                beta_weights[i, j] = coverage_ratio * offset_penalties[j]

        return beta_weights

    def compute_loss(self, outputs_bkd, batch_labels_bkd, beta_weights, triggers,perturbations, delta_t_list):
        """Compute loss with dynamic mask adjustment."""
        loss_tp, loss_clean = 0.0, 0.0  # target-pattern loss and clean-data loss

        for i in range(self.pattern_len):
            pred = outputs_bkd[i]  # prediction for current sliding window
            target = batch_labels_bkd[i]  # target for current sliding window

            # 1. dynamically build time mask for current window i
            var_time_masks = self.build_time_masks(delta_t_list, i)

            tp_mask = torch.zeros(self.fct_output_len, pred.shape[-1], device=self.device)  # same shape as pred

            # iterate over all attacked variables
            for idx, var in enumerate(self.atk_vars):
                var_idx = var.item()
                tp_mask[:, var_idx] = var_time_masks[var_idx].squeeze(0)

            clean_mask = 1.0 - tp_mask  # non-attack region

            # 5. compute attack-region loss (with beta weights), use MAE not MSE
            for idx, var in enumerate(self.atk_vars):
                var_idx = var.item()
                single_tp_mask = tp_mask[:, var_idx]  # [fct_output_len]
                single_error = torch.abs(pred[:, var_idx] - target[:, var_idx]) * single_tp_mask  # MAE loss
                single_count = torch.sum(single_tp_mask) + 1e-8
                loss_tp += beta_weights[i][idx] * (torch.sum(single_error) / single_count)

            # 6. compute clean-region loss (MAE)
            clean_error = torch.sum(clean_mask * torch.abs(pred - target))
            clean_count = torch.sum(clean_mask) + 1e-8
            loss_clean += clean_error / clean_count

        # 7. Soft clipping regularizer (per-variable clip to mu +/- sigma)
        # perturbations: [batch, num_atk_vars, T]
        mean = self.atk_vars_mean.view(1, -1, 1).to(self.device)  # shape [1, C, 1]
        std = self.atk_vars_std.view(1, -1, 1).to(self.device)  # shape [1, C, 1]

        overflow = torch.abs(triggers - mean) - std
        penalty = F.relu(overflow / std)
        #loss_clip = (penalty ** 2).mean()
       

        loss_norm = torch.abs(torch.sum(perturbations, dim=1)).mean()
        total_variation_loss=torch.mean(torch.abs(triggers[:, :, 1:] - triggers[:, :, :-1]))


        loss = (self.lambda_attack*loss_tp / self.pattern_len) + (self.lambda_clean * loss_clean / self.pattern_len) 
        + self.lam_norm*loss_norm-total_variation_loss
        ret_str = f'Final Loss: {loss.item()} Loss TP: {loss_tp.item()}, Loss Clean: {loss_clean.item()}, loss_norm: {loss_norm.item()},total_variation_loss:{total_variation_loss}'
        

        return loss, ret_str