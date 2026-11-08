"""
CATCH defense entrypoint parallel to run_gdn_defense.py.

**Same as GDN (stage 1):** identical YAMLs, ``poison_data`` (attacker + victim forward pass),
and ``prepare_gdn_data`` (normal / poisoned windows and labels).

**Different (stage 2):** exports TAB OTB CSV and runs the TAB benchmark with ``catch.CATCH``
only — no GDN graph, no GDN ``train()``. Two anomaly models, one shared poison pipeline.
"""
from __future__ import annotations

import os
import random
import sys

import click
import numpy as np
import torch
import yaml

_REPO = os.path.abspath(os.path.dirname(__file__))


def _seed_everything(seed: int = 1, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


@click.command(context_settings={"show_default": True})
@click.option(
    "--train-config-path",
    type=str,
    default=os.path.join(_REPO, "configs/backdoor_config/original_autoformer_cone.yaml"),
    help="Same as run_gdn_defense.py: Train yaml.",
)
@click.option(
    "--config-path",
    type=str,
    default=os.path.join(_REPO, "configs/surrogate_config/default_config.yaml"),
    help="Same as run_gdn_defense.py: surrogate yaml.",
)
@click.option("--batch-size", type=int, default=16, help="GDN-only; ignored for CATCH.")
@click.option("--epoch", type=int, default=100, help="GDN-only; ignored for CATCH.")
@click.option("--slide-win", type=int, default=1, help="GDN-only; ignored for CATCH.")
@click.option("--slide-stride", type=int, default=1, help="GDN-only; ignored for CATCH.")
@click.option("--dim", type=int, default=32, help="GDN-only; ignored for CATCH.")
@click.option("--out-layer-num", type=int, default=1, help="GDN-only; ignored for CATCH.")
@click.option("--out-layer-inter-dim", type=int, default=32, help="GDN-only; ignored for CATCH.")
@click.option("--topk", type=int, default=5, help="GDN-only; ignored for CATCH.")
@click.option("--val-ratio", type=float, default=0.1, help="GDN-only; ignored for CATCH.")
@click.option("--decay", type=float, default=0, help="GDN-only; ignored for CATCH.")
@click.option("--random-seed", type=int, default=5, help="Same as run_gdn_defense.py.")
@click.option("--report", type=str, default="best", help="GDN-only; ignored for CATCH.")
@click.option("--save-model-path", type=str, default=None, help="GDN-only; ignored for CATCH.")
@click.option("--load-model-path", type=str, default="", help="GDN-only; ignored for CATCH.")
@click.option(
    "--deterministic",
    type=bool,
    default=False,
    help="Same as run_gdn_defense.py (cudnn deterministic).",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    help="Passed to TAB run_benchmark --num-workers (GDN default 64 is not used here).",
)
@click.option("--max-timesteps", type=int, default=None, help="Optional truncate after export.")
@click.option("--skip-export", is_flag=True, help="Reuse existing CSV under CATCH dataset/.")
@click.option("--catch-hparams-json", type=str, default=None, help="Merge into CATCH hyperparameters.")
@click.option("--gpus", type=str, default=None, help='Override Train gpuid, e.g. "0".')
@click.option("--timeout", type=float, default=60000.0)
@click.option("--save-path", type=str, default=None, help="TAB result save-path (default label/CATCH_<dataset>_poison).")
@click.option("--out-name", type=str, default=None, help="Override CSV basename.")
@click.option("--python", type=str, default=sys.executable)
def main(
    train_config_path: str,
    config_path: str,
    batch_size: int,
    epoch: int,
    slide_win: int,
    slide_stride: int,
    dim: int,
    out_layer_num: int,
    out_layer_inter_dim: int,
    topk: int,
    val_ratio: float,
    decay: float,
    random_seed: int,
    report: str,
    save_model_path: str | None,
    load_model_path: str,
    deterministic: bool,
    num_workers: int,
    max_timesteps: int | None,
    skip_export: bool,
    catch_hparams_json: str | None,
    gpus: str | None,
    timeout: float,
    save_path: str | None,
    out_name: str | None,
    python: str,
) -> None:
    _ = (
        batch_size,
        epoch,
        slide_win,
        slide_stride,
        dim,
        out_layer_num,
        out_layer_inter_dim,
        topk,
        val_ratio,
        decay,
        report,
        save_model_path,
        load_model_path,
    )

    os.chdir(_REPO)
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    _seed_everything(random_seed, deterministic=deterministic)
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    tcp = train_config_path if os.path.isabs(train_config_path) else os.path.join(_REPO, train_config_path)
    cp = config_path if os.path.isabs(config_path) else os.path.join(_REPO, config_path)

    with open(tcp, "r") as f:
        train_cfg = yaml.safe_load(f)["Train"]
    dataset_name = train_cfg["dataset"]

    csv_name = out_name or f"{dataset_name}_catch_poison.csv"

    if not skip_export:
        import importlib.util

        exp_path = os.path.join(_REPO, "defense_models", "usad", "export_pems_for_catch.py")
        spec = importlib.util.spec_from_file_location("export_pems_for_catch", exp_path)
        if spec is None or spec.loader is None:
            raise ImportError(exp_path)
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
        exp.run_export_from_poison_pipeline(
            repo_root=_REPO,
            train_config=tcp,
            surrogate_config=cp,
            out_name=csv_name,
            max_timesteps=max_timesteps,
        )

    from run_catch_benchmark import run_catch_tab_eval

    code = run_catch_tab_eval(
        repo_root=_REPO,
        train_config_path=tcp,
        out_name=csv_name,
        catch_hparams_json=catch_hparams_json,
        gpus=gpus,
        num_workers=num_workers,
        timeout=timeout,
        save_path=save_path or f"label/CATCH_{dataset_name}_poison",
        python_exe=python,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
