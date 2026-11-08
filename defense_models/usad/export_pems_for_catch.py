"""
Export PEMS multivariate series (same source as prepare_dataset.py / USAD defense) into
TAB/CATCH OTB long CSV + DETECT_META.csv under defense_models/CATCH/dataset/anomaly_detect/.

Data file: by default uses Dataset.PEMS03 in --surrogate-config, i.e. data/PEMS03/PEMS03.npz
(relative to repo root). Override with --data-npz.

Labels: default --label-mode synthetic_event adds one synthetic anomaly segment in the test split
so CATCH affiliation metrics do not fail on empty ground-truth events.

Use run_export_from_poison_pipeline() to mirror run_gdn_defense.py: load attacker checkpoint,
build normal + poisoned train windows, then the same prepare_gdn_data() layout for CATCH CSV.

OTB layout (multivariate, no date): variable-major blocks; col0=value, col1=sensor id, col2=cols name.
See ts_benchmark.data.utils.process_data_df.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml


def _parser_args(config_path: str, train_config_path: str) -> dict:
    default_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)["Train"]
    config["Dataset"] = default_config["Dataset"][config["dataset"]]
    config["Target_Pattern"] = default_config["Target_Pattern"][config["pattern_type"]]
    config["Model"] = default_config["Model"][config["model_name"]]
    config["Model"]["c_out"] = config["Dataset"]["num_of_vertices"]
    config["Model"]["enc_in"] = config["Dataset"]["num_of_vertices"]
    config["Model"]["dec_in"] = config["Dataset"]["num_of_vertices"]
    config["Surrogate"] = default_config["Model"][config["surrogate_name"]]
    config["Surrogate"]["c_out"] = config["Dataset"]["num_of_vertices"]
    config["Surrogate"]["enc_in"] = config["Dataset"]["num_of_vertices"]
    config["Surrogate"]["dec_in"] = config["Dataset"]["num_of_vertices"]
    config["Trigger_Config"] = default_config["trigger_config"]
    return config


def _load_pems_train_test_flat(npz_path: str) -> np.ndarray:
    """Same 60% train / 20% val gap / 20% test split as dataset.load_raw_data for PEMS."""
    raw_data = np.load(npz_path)["data"]
    train_seq = raw_data[: int(0.6 * raw_data.shape[0])]
    test_seq = raw_data[int(0.8 * raw_data.shape[0]) :]
    train_flat = train_seq[:, :, 0].astype(np.float64)
    test_flat = test_seq[:, :, 0].astype(np.float64)
    return np.concatenate([train_flat, test_flat], axis=0)


def _build_otb_rows(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """values: (T, N), labels: (T,) binary/point anomaly."""
    T, N = values.shape
    names = [f"s{i}" for i in range(N)] + ["label"]
    rows = []
    for j, name in enumerate(names):
        for t in range(T):
            if name != "label":
                v = float(values[t, j])
            else:
                v = float(labels[t])
            rows.append([v, j, name])
    return pd.DataFrame(rows, columns=["data", "sid", "cols"])


def _persist_catch_csv(
    repo_root: str, out_name: str, df_long: pd.DataFrame, length: int, train_lens: int
) -> tuple[str, str]:
    catch_root = os.path.join(repo_root, "defense_models", "CATCH")
    ad_root = os.path.join(catch_root, "dataset", "anomaly_detect")
    data_dir = os.path.join(ad_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, out_name)
    df_long.to_csv(out_path, index=False)

    meta_path = os.path.join(ad_root, "DETECT_META.csv")
    row = {
        "file_name": out_name,
        "freq": "other",
        "if_univariate": False,
        "size": "large",
        "length": length,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": train_lens,
    }
    new_meta = pd.DataFrame([row])
    if os.path.isfile(meta_path):
        old = pd.read_csv(meta_path)
        old = old[old["file_name"] != out_name]
        meta = pd.concat([old, new_meta], ignore_index=True)
    else:
        meta = new_meta
    meta.to_csv(meta_path, index=False)
    print(f"[ok] wrote {out_path} ({len(df_long)} rows), train_lens={train_lens}, T={length}")
    print(f"[ok] updated {meta_path}")
    return os.path.abspath(out_path), os.path.abspath(meta_path)


def run_export_from_poison_pipeline(
    repo_root: str,
    train_config: str,
    surrogate_config: str,
    out_name: str,
    max_timesteps: int | None = None,
) -> tuple[str, str]:
    """
    Same data path as run_gdn_defense.py: poison_data() + prepare_gdn_data(), then OTB export.

    ``prepare_gdn_data`` receives ``device=torch.device('cuda')`` when CUDA is available,
    matching run_gdn_defense.py (GPU reshape path); otherwise ``None`` like GDN on CPU.

    train_lens = len(train_df) (normal-only segment). Test tail is normal+attack rows with
    attack column as point labels (same simplification as prepare_gdn_data).
    """
    if not os.path.isabs(train_config):
        train_config = os.path.join(repo_root, train_config)
    if not os.path.isabs(surrogate_config):
        surrogate_config = os.path.join(repo_root, surrogate_config)

    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        gdn_path = os.path.join(repo_root, "run_gdn_defense.py")
        spec = importlib.util.spec_from_file_location("run_gdn_defense_mod", gdn_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {gdn_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        normal, atk_train_set, time_to_attack = mod.poison_data(train_config, surrogate_config)
        if len(normal.shape) == 3:
            num_features = int(normal.shape[2])
        else:
            num_features = int(normal.shape[-1])

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        prep_device = torch.device("cuda") if use_cuda else None
        train_df, test_df, _feature_names = mod.prepare_gdn_data(
            normal, atk_train_set, time_to_attack, num_features, device=prep_device
        )
        feat_cols = [c for c in train_df.columns if c != "attack"]
        values = np.vstack(
            [train_df[feat_cols].values.astype(np.float64), test_df[feat_cols].values.astype(np.float64)]
        )
        labels = np.concatenate(
            [
                np.zeros(len(train_df), dtype=np.float64),
                test_df["attack"].values.astype(np.float64),
            ]
        )
        train_lens = int(len(train_df))

        if max_timesteps is not None and values.shape[0] > max_timesteps:
            values = values[:max_timesteps]
            labels = labels[:max_timesteps]
            train_lens = min(train_lens, max(1, max_timesteps - 1))

        if float(np.sum(labels)) < 1.0:
            raise RuntimeError(
                "Poison export produced no positive labels; cannot run affiliation metrics. "
                "Check attack data / checkpoint."
            )

        T, N = values.shape
        df_long = _build_otb_rows(values, labels)
        return _persist_catch_csv(repo_root, out_name, df_long, T, train_lens)
    finally:
        os.chdir(cwd)


def run_export(
    repo_root: str,
    train_config: str,
    surrogate_config: str,
    out_name: str,
    train_fraction: float = 0.6,
    max_timesteps: int | None = None,
    label_mode: str = "synthetic_event",
    data_npz: str | None = None,
) -> tuple[str, str]:
    """
    Export OTB CSV + DETECT_META row. Returns (absolute_csv_path, absolute_meta_path).
    """
    if not os.path.isabs(train_config):
        train_config = os.path.join(repo_root, train_config)
    if not os.path.isabs(surrogate_config):
        surrogate_config = os.path.join(repo_root, surrogate_config)

    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        config = _parser_args(surrogate_config, train_config)
        ds = config["Dataset"]
        if data_npz is not None:
            npz_path = data_npz
            if not os.path.isabs(npz_path):
                npz_path = os.path.join(repo_root, npz_path)
        else:
            rel = ds["data_filename"]
            if isinstance(rel, str) and rel.startswith("./"):
                rel = rel[2:]
            npz_path = rel if os.path.isabs(rel) else os.path.join(repo_root, rel)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"PEMS npz not found: {npz_path}")
        print(f"[info] loading {npz_path}")

        values = _load_pems_train_test_flat(npz_path)
        if max_timesteps is not None:
            values = values[:max_timesteps]

        T, N = values.shape
        train_lens = int(train_fraction * T)
        labels = np.zeros(T, dtype=np.float64)
        test_region = np.arange(train_lens, T)
        if label_mode == "zeros":
            pass
        elif label_mode == "synthetic_event":
            if len(test_region) > 0:
                span = min(96, max(8, len(test_region) // 20))
                start = train_lens + max(0, (len(test_region) - span) // 2)
                end = min(T, start + span)
                labels[start:end] = 1.0
        else:
            if len(test_region) > 0:
                rng = np.random.default_rng(0)
                k = max(1, len(test_region) // 200)
                pick = rng.choice(test_region, size=min(k, len(test_region)), replace=False)
                labels[pick] = 1.0

        df_long = _build_otb_rows(values, labels)
        return _persist_catch_csv(repo_root, out_name, df_long, T, train_lens)
    finally:
        os.chdir(cwd)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-config",
        default="configs/backdoor_config/original_autoformer_cone.yaml",
    )
    ap.add_argument(
        "--surrogate-config",
        default="configs/surrogate_config/default_config.yaml",
    )
    ap.add_argument(
        "--out-name",
        default="PEMS03_usad.csv",
        help="Filename under CATCH dataset/anomaly_detect/data/",
    )
    ap.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="train_lens = floor(train_fraction * T); matches load_raw_data split.",
    )
    ap.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional cap on T (from start) for faster experiments.",
    )
    ap.add_argument(
        "--label-mode",
        choices=("synthetic_event", "zeros", "random_test"),
        default="synthetic_event",
        help="synthetic_event (default): one contiguous 1-block in the test region so TAB/CATCH "
        "affiliation metrics run (need >=1 gt event). zeros: all-normal (many metrics become NaN). "
        "random_test: sparse point 1s in test region.",
    )
    ap.add_argument(
        "--data-npz",
        default=None,
        help="Path to PEMS03.npz (default: Dataset.data_filename from surrogate config, "
        "normally data/PEMS03/PEMS03.npz under repo root).",
    )
    ap.add_argument(
        "--from-poison",
        action="store_true",
        help="Use run_gdn_defense.poison_data + prepare_gdn_data (same as GDN defense) instead of raw npz.",
    )
    args = ap.parse_args()

    if args.from_poison:
        run_export_from_poison_pipeline(
            repo_root=repo_root,
            train_config=args.train_config,
            surrogate_config=args.surrogate_config,
            out_name=args.out_name,
            max_timesteps=args.max_timesteps,
        )
    else:
        run_export(
            repo_root=repo_root,
            train_config=args.train_config,
            surrogate_config=args.surrogate_config,
            out_name=args.out_name,
            train_fraction=args.train_fraction,
            max_timesteps=args.max_timesteps,
            label_mode=args.label_mode,
            data_npz=args.data_npz,
        )


if __name__ == "__main__":
    main()
