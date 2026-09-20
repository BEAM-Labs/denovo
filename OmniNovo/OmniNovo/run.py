"""The command line entry point for OmniNovo."""
from loguru import logger
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import click
import torch
import yaml
from pytorch_lightning.lite import LightningLite

from . import utils
from .train import model_runner


@click.command()
@click.option(
    "--mode",
    required=True,
    default="denovo",
    type=click.Choice(["denovo"]),
)
@click.option(
    "--model",
    help="The file name of the model weights (.ckpt file).",
)
@click.option(
    "--input_path",
    required=True,
    help="Input spectra path for inference",
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options. If not "
    "specified, a default configuration will be used.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="Base output file name for prediction results.",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--log_base",
    help="Folder for prediction TSV output.",
)
def main(
    mode: str,
    model: Optional[str],
    input_path: str,
    config: Optional[str],
    output: Optional[str],
    log_base: Optional[str],
):
    peak_path = input_path
    if output is None:
        output = os.path.join(
            os.getcwd(),
            "logs",
            "predictions",
        )
    else:
        output = os.path.splitext(os.path.abspath(output))[0]
    if log_base is None:
        log_base = output

    # Read parameters from the config file.
    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "default_rules.yaml"
        )
    config_fn = config
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    # Ensure that the config values have the correct type.
    config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_length=int,
        n_log=int,
        predict_batch_size=int,
        n_beams=int,
    )
    for k, t in config_types.items():
        try:
            if config[k] is not None:
                config[k] = t(config[k])
        except (TypeError, ValueError) as e:
            logger.error("Incorrect type for configuration value %s: %s", k, e)
            raise TypeError(f"Incorrect type for configuration value {k}: {e}")
    config["residues"] = {
        str(aa): float(mass) for aa, mass in config["residues"].items()
    }
    config["n_workers"] = utils.n_workers()

    import random
    if(config["random_seed"]==-1):
        config["random_seed"]=random.randint(1, 9999)
    LightningLite.seed_everything(seed=config["random_seed"], workers=True)

    logger.info("mode = %s", mode)
    logger.info("model = %s", model)
    logger.info("input = %s", peak_path)
    logger.info("config = %s", config_fn)
    logger.info("output = %s", output)
    logger.info("Predict sequences.")
    model_runner.predict(peak_path, model, config, None, log_base)

if __name__ == "__main__":
    main()
