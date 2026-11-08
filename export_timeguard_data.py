"""Export an HSAT-repo attacker (Backtime / MSAT / HSAT) into TimeGuard's input format:
  <out_dir>/poisoned_dataset.csv    full series, train split (first 60%) replaced by poisoned data
  <out_dir>/test_attacked_data.pth  triggered test windows + target-pattern labels
Mirrors TimeGuard's attack_backtime_trainer.save_poisoning_data / dataset_attack.save_attacked_dataset.

Usage: python export_timeguard_data.py --config-path <surrogate cfg> --train-config-path <train cfg> --out-dir <dir>
"""
import glob, os
import click, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from run import parser_args, seed_everything
from dataset import load_raw_data, AttackEvaluateSet
from trainer import Trainer
from timefeatures import time_features


def find_checkpoint(config):
    tg = config.Trigger_Config.trigger_generator
    base = (f'attacker_{config.dataset}_{config.pattern_type}_{tg.type}_hidden_dim={tg.hidden_dim}'
            f'_atklr={config.attack_lr}_globalvslocalratio={tg.feature_alpha}_warmup={config.warmup}_{config.surrogate_name}')
    # same preference order as run_gdn_defense.py / run_usad_defense.py
    for f in [f'./checkpoints/transferlearning_{base}_clean=False.pth', f'./checkpoints/{base}.pth',
              f'./checkpoints/transferlearning_{base}.pth']:
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f'No attacker checkpoint for {base}')


@click.command()
@click.option('--config-path', required=True)
@click.option('--train-config-path', required=True)
@click.option('--out-dir', required=True)
def main(config_path, train_config_path, out_dir):
    config = parser_args(config_path, train_config_path)
    seed_everything(config.Trigger_Config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std, train_seq, test_seq = load_raw_data(config.Dataset)

    # atk_vars / target_pattern / atk_ts are overwritten by the checkpoint
    atk_vars = torch.arange(max(int(round(train_seq.shape[1] * config.alpha_s)), 1)).to(device)
    target_pattern = torch.tensor(config.Target_Pattern).float().to(device) * std
    trainer = Trainer(config, atk_vars, target_pattern, mean, std, train_seq, test_seq,
                      None, None, device, None, no_attack=False)
    ckpt = find_checkpoint(config)
    trainer.load_attacker(torch.load(ckpt, weights_only=False))
    attacker = trainer.attacker
    attacker.eval()
    assert attacker.atk_ts is not None and len(attacker.atk_ts) > 0, 'checkpoint has no atk_ts'
    print('[INFO] checkpoint:', ckpt)
    print(f'[INFO] atk_vars={len(attacker.atk_vars)} atk_ts={len(attacker.atk_ts)}')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # 1) poisoned training split -> full-series CSV
        attacker.sparse_inject()
        poison = attacker.dataset.poisoned_data[:, 0, :].cpu().numpy().T  # (T_train, n)
        if 'PEMS' in config.dataset:
            df = pd.DataFrame(np.load(config.Dataset.data_filename)['data'][:, :, 0])
            df.insert(0, 'date', None)
        else:
            df = pd.read_csv(config.Dataset.data_filename)
        assert poison.shape == (int(0.6 * len(df)), df.shape[1] - 1), (poison.shape, df.shape)
        n_changed = int((~np.isclose(df.iloc[:poison.shape[0], 1:].values.astype(np.float32), poison)).sum())
        df.iloc[:poison.shape[0], 1:] = poison
        df.to_csv(os.path.join(out_dir, 'poisoned_dataset.csv'), index=False)
        print(f'[INFO] poisoned_dataset.csv written, {n_changed} train values changed')

        # 2) triggered test windows
        atk_set = AttackEvaluateSet(attacker, test_seq, mean, std, device, num_for_hist=12, num_for_futr=12)
        loader = DataLoader(atk_set, batch_size=config.batch_size, shuffle=False, collate_fn=atk_set.collate_fn)
        feats, orig, targets, clean, idxs = [], [], [], [], []
        for f, t, c, i in loader:
            feats.append(atk_set.denormalize(f).cpu())  # TimeGuard re-normalizes with its own stats
            targets.append(t.cpu()); clean.append(c.cpu()); idxs.append(torch.as_tensor(i))
        for i in range(len(atk_set)):
            orig.append(atk_set.data[:, 0:1, i:i + 12].cpu())
        d = {'atk_vars': attacker.atk_vars.cpu(), 'atk_ts': attacker.atk_ts.cpu(),
             'trigger_len': torch.tensor(attacker.trigger_len), 'pattern_len': torch.tensor(attacker.pattern_len),
             'features': torch.cat(feats), 'original_features': torch.stack(orig), 'targets': torch.cat(targets),
             'clean_targets': torch.cat(clean), 'indices': torch.cat(idxs)}
        assert torch.equal(d['indices'], torch.arange(len(atk_set)))

        # TimeGuard runs Weather/ETTm1 with timestamps; indexes stamps by sample index
        if 'PEMS' not in config.dataset:
            dates = pd.to_datetime(df.iloc[int(0.8 * len(df)):, 0].values)
            st = torch.from_numpy(time_features(dates).T).float()  # (T_test, 4)
            n = len(atk_set)
            d['input_stamps'] = torch.stack([st[i:i + 12] for i in range(n)])
            d['target_stamps'] = torch.stack([st[i + 12:i + 24] for i in range(n)])

        torch.save(d, os.path.join(out_dir, 'test_attacked_data.pth'))
        print('[INFO] test_attacked_data.pth written:', {k: tuple(v.shape) for k, v in d.items() if v.dim()})


if __name__ == '__main__':
    main()
