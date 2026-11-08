# Structure-Aware Adaptive Triggers for Backdoor Attacks on Multivariate Time-Series Forecasting

Official code for **MSAT** and **HSAT**, two structure-aware adaptive trigger generators for backdoor attacks on multivariate time-series (MTS) forecasting.

- **MSAT** (`TgrCNN1D`): a 1D-CNN trigger generator (Inception blocks + CBAM attention) that learns local temporal structure per variable.
- **HSAT** (`TgrHybridGCN`): a hybrid trigger generator that blends local CNN features with global inter-variable graph (GCN) features (`feature_alpha` controls the blend).

Both plug into the bi-level surrogate/trigger optimization of [BackTime](https://arxiv.org/abs/2410.02195), which is the main baseline. Victim models: Autoformer, FEDformer, TimesNet. Datasets: PEMS03, PEMS04, PEMS08, Weather, ETTm1.

This README explains how to reproduce every experiment: the attacks (Clean, Backtime, MSAT, HSAT, TDBA), the detectability study (GDN, USAD, CATCH) and the mitigation study (TimeGuard). All commands run from the repo root unless stated otherwise.

---
## Abstract
Multivariate time-series forecasting models can be vulnerable to training-time backdoor attacks that preserve clean forecasting performance while inducing targeted deviations on triggered inputs. We investigate whether temporal and relational inductive biases improve adaptive trigger generation. We propose a Multi-Scale Adaptive Trigger (MSAT), which processes variable histories at multiple temporal scales, and a Hybrid-Structured Adaptive Trigger (HSAT), which combines temporal processing with a learned variable graph. We evaluate both generators on five datasets and three forecasting architectures, with additional target-pattern and cross-architecture experiments. Performance depends on the dataset and evaluation setting: HSAT performs favorably on several traffic benchmarks, whereas MSAT is competitive or stronger in others. Results under TimeGuard are also dataset-dependent. A conditional fixed-forecaster analysis characterizes when complementary trigger branches can reduce feature-space misalignment. These findings motivate reporting targeted-support error and per-variable behavior alongside aggregate forecasting accuracy.

## 0. Setup

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
export TORCHDYNAMO_DISABLE=1   # avoids a pkg_resources crash in torch inductor with recent setuptools
```

Data layout expected by `configs/surrogate_config/*.yaml`:

```
data/PEMS03/PEMS03.npz
data/PEMS04/PEMS04.npz
data/PEMS08/PEMS08.npz
data/all_datasets/weather/weather.csv
data/all_datasets/ETT-small/ETTm1.csv
```

(`data/all_datasets.zip` contains Weather and ETT; unzip it into `data/`.)

### Shared-GPU etiquette

When launching many runs in parallel, **cap CPU threads**:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Without it every PyTorch process spawns ~64 threads. 30 parallel TimeGuard runs pushed load to ~790 on 64 cores, and it ran 20–100× slower than with the cap.

---

## 1. Methods and their configs

Every attack run is `run.py` with two configs:

- `--config-path`: the **surrogate config**, which selects the method / trigger generator.
- `--train-config-path`: the **backdoor config**, which selects the dataset, victim model and target pattern.

| Method | Surrogate config (`configs/surrogate_config/`) | Trigger generator |
|---|---|---|
| Clean (no attack) | `clean_config.yaml` | – |
| Backtime (baseline) | `default_config.yaml` | `TgrGCN` |
| **MSAT** (ours) | `surrogate_config_TgrCNN1D.yaml` | `TgrCNN1D` |
| **HSAT** (ours) | `surrogate_config_TgrGLGCN.yaml` | `TgrHybridGCN` (feature_alpha 0.9) |

Backdoor configs used for the main tables: `configs/backdoor_config/full_table1_{DATASET}_{MODEL}_cone.yaml`, where DATASET ∈ {PEMS03, PEMS04, PEMS08, Weather, ETTm1} and MODEL ∈ {Autoformer, FEDformer, TimesNet}.

**Seed:** the attack seed is `Trigger_Config.seed` inside the surrogate config (default `5`). To run another seed, edit that field in all 4 surrogate configs.

---

## 2. Attacks (Table 1)

One run:

```bash
python run.py \
  --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml \
  --train-config-path configs/backdoor_config/full_table1_PEMS03_Autoformer_cone.yaml
```

Each run has two stages:

1. **Train the surrogate and trigger generator.** The result is saved to `checkpoints/transferlearning_attacker_{ds}_{pattern}_{TgrType}_hidden_dim=..._atklr=..._globalvslocalratio=..._warmup=..._{surrogate}_clean={True|False}.pth`. If that file already exists it is **loaded and training is skipped**. Delete it to retrain.
2. **Train a fresh victim model** on the poisoned data and report clean and attacked MAE/RMSE. Metrics go to `log_results/seed{seed}/{ds}_{pattern}_{TgrType}_{surrogate}_{model}/*_test_metrics.csv` (Clean runs use `log_results/seed{seed}/CLEAN_...`).

Full Table 1 (5 datasets × Clean/Backtime/MSAT/HSAT, one script per victim model):

```bash
bash scripts/Table1_FULL_Autoformer_submit_job.sh
bash scripts/Table1_FULL_FEDformer_submit_job.sh
bash scripts/Table1_FULL_TimesNet_submit_job.sh
```

Other tables:

```bash
bash scripts/Table3_submit_job.sh   # target patterns: up-trend, up-and-down
bash scripts/Table4_submit_job.sh   # transfer: FEDformer surrogate -> TimesNet victim
```

### TDBA baseline

TDBA has its own code and venv in `TDBA/`:

```bash
cd TDBA
TDBA_SEED=5 CUDA_VISIBLE_DEVICES=0 python3 main.py cone
```

`TDBA/run_tdba_run3_final.sh` launches Autoformer/FEDformer/TimesNet in tmux on GPUs 0/1/2. Results are appended to `TDBA/result_XXXX_{pattern}.txt`. Runs append to the same file, so de-duplicate before reading.

---

## 3. Detectability: GDN and USAD (Table 2)

Both need the **attacker checkpoint from step 2** (the `_clean=False` one). They rebuild the poisoned data from it and report **F1 and AUC** for detecting triggered timestamps; lower means stealthier.

Runner: `scripts/run_gdn_usad.sh <gdn|usad> <gpu> <par|seq> DATASET...` (runs Backtime, MSAT and HSAT for each dataset, using the `full_table1_{ds}_Autoformer_cone.yaml` backdoor config).

```bash
bash scripts/run_gdn_usad.sh gdn  0 par Weather ETTm1 PEMS04 PEMS08   # par = 3 triggers in parallel
bash scripts/run_gdn_usad.sh usad 1 par Weather ETTm1 PEMS04 PEMS08
bash scripts/run_gdn_usad.sh usad 2 seq PEMS03                        # see memory note below
```

Logs: `logs/table2_weather/{gdn|usad}_{backtime|msat|hsat}_{ds}.log`. The last lines contain:

- GDN: `F1 score: ...` / `AUC score: ...`
- USAD: `f1= ...` / `auc_score= ...`

Collected values are in `logs/table2_weather/results.csv`. PEMS is reported as the mean over PEMS03/04/08.

Equivalent direct commands (what `run_gdn_usad.sh` runs):

```bash
python run_gdn_defense.py --train-config-path configs/backdoor_config/full_table1_PEMS03_Autoformer_cone.yaml \
  --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml \
  --epoch 50 --num-workers 4 --batch-size 32 --out-layer-inter-dim 64 --decay 0.05 \
  --out-layer-num 7 --val-ratio 0.1 --dim 32 --random-seed 5 --save-model-path tmp/gdn_hsat_PEMS03.pt

python run_usad_defense.py --train-config-path configs/backdoor_config/full_table1_PEMS03_Autoformer_cone.yaml \
  --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml --alpha-loss1 0.5 --random-seed 5
```

Notes:
- **USAD on PEMS03** needs about 35 GB of GPU memory (~1.7B parameters). Run it alone on an empty 40 GB GPU.
- **USAD threshold:** USAD uses a fixed 0.5 threshold on raw scores. F1 = 0 with AUC = 0.5 means nothing was flagged.
- **Seed:** `--random-seed` controls the detector seed. Our reported numbers use seed 5 only.

### CATCH (optional detector)

```bash
python run_catch_defense.py \
  --train-config-path configs/backdoor_config/full_table1_PEMS03_Autoformer_cone.yaml \
  --config-path configs/surrogate_config/surrogate_config_TgrCNN1D.yaml \
  --gpus 0 --num-workers 30 --batch-size 16 --epoch 1 --deterministic True
```

See `run_defense_catch.sh` for all three triggers.

---

## 4. Mitigation: TimeGuard

TimeGuard lives in a separate repo: `~/BadFedBacktime/defense_models/TimeGuard` (with its own `.venv`, Python 3.11). It retrains a forecaster on poisoned data while filtering suspicious samples. It reports:

- **clean MAE**: MAE on the clean test set.
- **attack MAE**: MAE between predictions on triggered test inputs and the target pattern. **Higher means the defense removed more of the backdoor**; lower means the attack survived better.

### 4.1 Export HSAT-repo attacks into TimeGuard's format

Run from **this** repo, once per (dataset, trigger). It needs the attacker checkpoint from step 2.

```bash
TG=~/BadFedBacktime/defense_models/TimeGuard
python export_timeguard_data.py \
  --config-path configs/surrogate_config/surrogate_config_TgrGLGCN.yaml \
  --train-config-path configs/backdoor_config/full_table1_PEMS03_Autoformer_cone.yaml \
  --out-dir $TG/checkpoints/PEMS03_hsat_Autoformer_1212
```

The export writes two files into `--out-dir`:

- `poisoned_dataset.csv`: the full series, with the training split (first 60%) replaced by the poisoned data.
- `test_attacked_data.pth`: triggered test windows, target-pattern labels, `atk_vars` and `atk_ts`.

Use trigger names `backtime` / `msat` / `hsat` in the folder name with the matching surrogate config from section 1.

### 4.2 TimeGuard config

One per (dataset, trigger): `$TG/configs/timeguard/{ds}_{trigger}_Autoformer_1212/Autoformer/TimeGuard.yaml`. All 15 already exist. For a new one, copy an existing file and change `dataset` and `attack_save_folder`. Defaults: alpha 0.2, beta 0.5, t_b 10, t_1 10, t_2 90, k_nn 20, pi 1.25, lr 1e-4, batch 64, seq/pred 12.

Datasets are registered in `$TG/configs/default_config.yaml` (PEMS03/04/08, Weather, ETTm1).

### 4.3 Run

```bash
cd ~/BadFedBacktime/defense_models/TimeGuard
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python defense_timeguard.py --seed 1 \
  --defense_config_path configs/timeguard/PEMS03_hsat_Autoformer_1212/Autoformer/TimeGuard.yaml \
  > logs/tg_PEMS03_hsat.log 2>&1
```

All 30 runs (5 datasets × 3 triggers × seeds 1, 2) in parallel on 4 GPUs. Edit the GPU mapping inside the script to fit free memory (~1.5 GB per run):

```bash
bash logs/launch_all.sh
```

This takes about 6 h on 4 × A100 when the CPU threads are capped; ETTm1 is the slowest.

The last line of each log is `Final results: {'cln_mae': ..., 'atk_mae': ...}`. Seed 1 logs are `logs/tg_{ds}_{trigger}.log`; seed 2 logs are `logs/tg_{ds}_{trigger}_seed2.log`.

### 4.4 Aggregate

```bash
.venv/bin/python logs/collect_tg.py   # -> logs/timeguard_results.csv
```

The script reports mean ± std over the seeds (population std). PEMS = per-seed mean over PEMS03/04/08, then mean ± std over seeds. It also lists any missing runs.

Caveats:
- **Epoch selection uses the test set:** TimeGuard picks the best epoch by test clean MAE.
- **Normalization:** it normalizes with statistics of the poisoned training split.
- **Std meaning:** the poisoned data is the same for both seeds, so the std reflects TimeGuard's own randomness only.

---

## 5. Where results live

| Experiment | Output |
|---|---|
| Attacks (Table 1/3/4) | `log_results/seed{seed}/.../*_test_metrics.csv` |
| Attacker checkpoints | `checkpoints/transferlearning_attacker_*.pth` |
| GDN / USAD | `logs/table2_weather/*.log`, `logs/table2_weather/results.csv` |
| TimeGuard | `$TG/logs/tg_*.log`, `$TG/logs/timeguard_results.csv` |
| TDBA | `TDBA/result_XXXX_{pattern}.txt` |

---

## Acknowledgements

Built on [BackTime](https://github.com/xiaolin-cs/BackTime) (Lin et al., 2024). The baselines and defenses come from their original authors: TDBA (`TDBA/`), GDN (`defense_models/GDN`), USAD (`defense_models/usad`), CATCH (`defense_models/CATCH`) and TimeGuard.
