"""
Run TAB/CATCH on multivariate series for the same YAML pair as run_gdn_defense.

**Default (recommended):** same poison pipeline as ``run_gdn_defense.py`` — attacker checkpoint,
``poison_data`` + ``prepare_gdn_data``, then TAB ``catch.CATCH``. The victim forecaster is only
used *inside* ``poison_data`` (as in GDN), not as the TAB model.

**``--raw-npz-only``:** export from PEMS npz splits only (no attacker), for quick baselines.

Prefer ``run_catch_defense.py`` for a Click CLI that mirrors ``run_gdn_defense.py`` flag-for-flag
(GDN-only flags are accepted but ignored; CATCH runs after the shared poison stage).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# Same hyperparameters as defense_models/CATCH/.../MSL_script/CATCH.sh (MSL baseline).
_DEFAULT_CATCH_HPARAMS: dict = {
    "Mlr": 1e-05,
    "auxi_lambda": 0.05,
    "batch_size": 8,
    "cf_dim": 32,
    "d_ff": 256,
    "d_model": 32,
    "dc_lambda": 0.1,
    "dropout": 0.05,
    "e_layers": 3,
    "head_dim": 32,
    "head_dropout": 0.1,
    "lr": 0.0005,
    "n_heads": 1,
    "num_epochs": 5,
    "patch_size": 16,
    "patch_stride": 8,
    "score_lambda": 0.05,
    "seq_len": 192,
    "anomaly_ratio": 1.0,
}


def _gpus_from_train(train: dict) -> list[str]:
    raw = train.get("gpuid", "0")
    if isinstance(raw, int):
        return [str(raw)]
    s = str(raw).replace("'", "").strip()
    return [x for x in s.replace(",", " ").split() if x] or ["0"]


def run_catch_tab_eval(
    repo_root: str,
    train_config_path: str,
    out_name: str,
    catch_hparams_json: str | None,
    gpus: str | None,
    num_workers: int,
    timeout: float,
    save_path: str,
    python_exe: str,
) -> int:
    """
    Run CATCH TAB benchmark on an existing OTB CSV under CATCH dataset/anomaly_detect/data/.
    Returns subprocess exit code (0 = success).
    """
    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)["Train"]

    hparams = dict(_DEFAULT_CATCH_HPARAMS)
    if catch_hparams_json:
        with open(catch_hparams_json, "r") as f:
            hparams.update(json.load(f))

    if gpus is not None:
        gpu_list = [x for x in gpus.replace(",", " ").split() if x]
    else:
        gpu_list = _gpus_from_train(train_cfg)

    catch_root = os.path.join(repo_root, "defense_models", "CATCH")
    csv_path = os.path.join(catch_root, "dataset", "anomaly_detect", "data", out_name)
    if not os.path.isfile(csv_path):
        print(f"error: missing {csv_path}", file=sys.stderr)
        return 1

    rb = os.path.join(catch_root, "scripts", "run_benchmark.py")
    cmd = [
        python_exe,
        rb,
        "--config-path",
        "unfixed_detect_label_multi_config.json",
        "--data-name-list",
        out_name,
        "--model-name",
        "catch.CATCH",
        "--model-hyper-params",
        json.dumps(hparams),
        "--gpus",
        *gpu_list,
        "--num-workers",
        str(num_workers),
        "--timeout",
        str(timeout),
        "--save-path",
        save_path,
    ]
    print("[info]", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=catch_root).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CATCH benchmark with config paths aligned to run_gdn_defense / run_usad_defense"
    )
    ap.add_argument(
        "--train-config-path",
        type=str,
        default=os.path.join(_REPO_ROOT, "configs/backdoor_config/original_autoformer_cone.yaml"),
        help="Same as run_gdn_defense.py: Train yaml (dataset, gpuid, …).",
    )
    ap.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(_REPO_ROOT, "configs/surrogate_config/default_config.yaml"),
        help="Same as run_gdn_defense.py: surrogate yaml.",
    )
    ap.add_argument(
        "--raw-npz-only",
        action="store_true",
        help="Export from npz only (no attacker). Default is GDN-aligned poison_data + prepare_gdn_data.",
    )
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip CSV generation; file must already exist under CATCH dataset/anomaly_detect/data/.",
    )
    ap.add_argument("--max-timesteps", type=int, default=None, help="Truncate exported series length.")
    ap.add_argument(
        "--label-mode",
        default="synthetic_event",
        choices=("synthetic_event", "zeros", "random_test"),
        help="Only for --raw-npz-only export.",
    )
    ap.add_argument("--train-fraction", type=float, default=0.6, help="Only for raw-npz export.")
    ap.add_argument("--data-npz", type=str, default=None, help="Override npz path for raw export.")
    ap.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Override output CSV basename (default: <dataset>_catch_poison.csv or <dataset>_usad.csv for raw).",
    )
    ap.add_argument(
        "--catch-hparams-json",
        type=str,
        default=None,
        help="Optional JSON file merged on top of default CATCH hyperparameters.",
    )
    ap.add_argument("--gpus", type=str, default=None, help='Override Train gpuid, e.g. "0".')
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60000.0)
    ap.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="CATCH result subpath (default: label/CATCH_<dataset>_poison or ..._usad if raw).",
    )
    ap.add_argument("--python", type=str, default=sys.executable, help="Interpreter for run_benchmark.py.")
    args = ap.parse_args()

    os.chdir(_REPO_ROOT)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    train_path = args.train_config_path
    if not os.path.isabs(train_path):
        train_path = os.path.join(_REPO_ROOT, train_path)
    sur_path = args.config_path
    if not os.path.isabs(sur_path):
        sur_path = os.path.join(_REPO_ROOT, sur_path)

    with open(train_path, "r") as f:
        train_cfg = yaml.safe_load(f)["Train"]
    dataset_name = train_cfg["dataset"]
    if "PEMS" not in str(dataset_name):
        print(
            f"error: export is only wired for PEMS datasets (dataset={dataset_name}).",
            file=sys.stderr,
        )
        sys.exit(2)

    use_poison = not args.raw_npz_only
    if args.out_name:
        out_name = args.out_name
    elif use_poison:
        out_name = f"{dataset_name}_catch_poison.csv"
    else:
        out_name = f"{dataset_name}_usad.csv"

    if not args.skip_export:
        _exp_path = os.path.join(_REPO_ROOT, "defense_models", "usad", "export_pems_for_catch.py")
        _spec = importlib.util.spec_from_file_location("export_pems_for_catch", _exp_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(_exp_path)
        _exp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_exp)

        if use_poison:
            _exp.run_export_from_poison_pipeline(
                repo_root=_REPO_ROOT,
                train_config=train_path,
                surrogate_config=sur_path,
                out_name=out_name,
                max_timesteps=args.max_timesteps,
            )
        else:
            _exp.run_export(
                repo_root=_REPO_ROOT,
                train_config=train_path,
                surrogate_config=sur_path,
                out_name=out_name,
                train_fraction=args.train_fraction,
                max_timesteps=args.max_timesteps,
                label_mode=args.label_mode,
                data_npz=args.data_npz,
            )

    if args.save_path:
        save_path = args.save_path
    elif use_poison:
        save_path = f"label/CATCH_{dataset_name}_poison"
    else:
        save_path = f"label/CATCH_{dataset_name}_usad"

    code = run_catch_tab_eval(
        repo_root=_REPO_ROOT,
        train_config_path=train_path,
        out_name=out_name,
        catch_hparams_json=args.catch_hparams_json,
        gpus=args.gpus,
        num_workers=args.num_workers,
        timeout=args.timeout,
        save_path=save_path,
        python_exe=args.python,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
