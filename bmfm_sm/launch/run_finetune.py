import logging
import os
import re
import sys
import warnings
from pathlib import Path

import click
from omegaconf import OmegaConf

# Needs to be set before PyTorch imports
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logging.info(
        f"Exported Darwin-specific env setting: {os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}"
    )

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

warnings.filterwarnings("ignore")

# Get BMFM_HOME environment variable
BMFM_HOME = os.environ.get("BMFM_HOME")
if not BMFM_HOME:
    logging.error("BMFM_HOME environment variable is not set.")
    sys.exit(1)

CONFIGS_DIR = os.path.join(BMFM_HOME, "configs_finetune")


class FinetuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--expt.name", type=str, required=False)
        parser.add_argument("--expt.dataset", type=str, required=True)
        parser.add_argument("--expt.split_strategy", type=str, required=True)
        parser.add_argument("--expt.model", type=str, required=True)
        parser.add_argument("--expt.ckpt", type=str, required=False, default=None)
        parser.add_argument("--expt.seed", type=int, required=False, default=None)
        parser.add_argument(
            "--expt.train_frac", type=float, required=False, default=None
        )
        parser.add_argument(
            "--tune_lr", required=False, default=None, action="store_true"
        )

    def before_instantiate_classes(self):
        logging.info(f"Running {self.subcommand} using configuration {self.config}")
        config = self.get_actual_config()

        if (
            self.subcommand != "test"
            and os.path.exists(os.path.dirname(config["ckpt_path"]))
            and "last" not in os.path.basename(config["ckpt_path"])
        ):
            ckpt_dir = os.path.dirname(config["ckpt_path"])
            ckpt_files = os.listdir(ckpt_dir)
            last_files = [
                os.path.join(ckpt_dir, ckpt_f)
                for ckpt_f in ckpt_files
                if "last" in ckpt_f
            ]
            last_files = [(f, os.path.getmtime(f)) for f in last_files]
            last_files = sorted(last_files, key=lambda x: x[1])
            if last_files:
                config["ckpt_path"] = last_files[-1][0]

        # Check if checkpoint exists and update config file
        ckpt_found = config.get("ckpt_path") and os.path.isfile(config["ckpt_path"])

        if not ckpt_found:
            if "subcommand" in self.config:
                self.config[self.config["subcommand"]]["ckpt_path"] = None
            else:
                self.config["ckpt_path"] = None
        else:
            logging.info(
                f"Found checkpoint mentioned in the configuration: {config['ckpt_path']}"
            )

        if self.subcommand == "test" and not ckpt_found:
            logging.info("Test was not run because finetuned checkpoint not found.")
            sys.exit()
        else:
            logging.info(config.get("expt.name"))

    def get_actual_config(self):
        config = (
            self.config[self.config["subcommand"]]
            if "subcommand" in self.config
            else self.config
        )
        return config

    def after_fit(self) -> None:
        logging.info("Training done. सुभां")


def get_available_options(path, option_type):
    """
    Returns a list of available options (directories or files) at the given path.

    :param path: Path object
    :param option_type: 'dir' for directories, 'file' for files
    :return: List of option names
    """
    if path.exists() and path.is_dir():
        if option_type == "dir":
            return [d.name for d in path.iterdir() if d.is_dir()]
        elif option_type == "file":
            return [
                f.name[len("config-") : -len(".yaml")]
                for f in path.iterdir()
                if f.is_file()
                and f.name.startswith("config-")
                and f.name.endswith(".yaml")
            ]
    return []


def select_option(option_name, provided_value, path, option_type="dir"):
    """
    Selects and validates an option.

    :param option_name: Name of the option (e.g., 'model')
    :param provided_value: Value provided by the user or None
    :param path: Path to search for available options
    :param option_type: 'dir' for directories, 'file' for files
    :return: Selected option value
    """
    options = get_available_options(path, option_type)
    if not options:
        logging.error(f"No {option_name}s found at {path}")
        sys.exit(1)
    if provided_value:
        if str(provided_value) not in options:
            logging.error(
                f"{option_name.capitalize()} '{provided_value}' is not available. "
                f"Available {option_name}s: {', '.join(options)}"
            )
            sys.exit(1)
        return str(provided_value)

    if len(options) == 1:
        logging.info(f"Using '{options[0]}' for {option_name}.")
        return options[0]

    return click.prompt(
        f"Please choose a {option_name}",
        type=click.Choice(options),
        show_choices=True,
    )


@click.command()
@click.option("--model", help="Model name")
@click.option("--dataset-group", help="Dataset group name")
@click.option("--split-strategy", help="Split strategy name")
@click.option("--dataset", help="Dataset name")
@click.option(
    "--fit", "mode", flag_value="fit", default=True, help="Run training (fit)"
)
@click.option("--test", "mode", flag_value="test", help="Run testing")
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="Override parameters in key=value format (e.g., trainer.max_epochs=10)",
)
def main(model, dataset_group, split_strategy, dataset, mode, override):
    base_path = Path(CONFIGS_DIR)
    model = select_option("model", model, base_path, "dir")
    dataset_group_path = base_path / model
    dataset_group = select_option(
        "dataset group", dataset_group, dataset_group_path, "dir"
    )
    split_strategy_path = dataset_group_path / dataset_group
    split_strategy = select_option(
        "split strategy", split_strategy, split_strategy_path, "dir"
    )
    dataset_path = split_strategy_path / split_strategy
    dataset = select_option("dataset", dataset, dataset_path, "dir")

    config_file = find_config_file(
        model, dataset_group, split_strategy, dataset, seed=101
    )
    if not os.path.exists(config_file):
        logging.error(f"Config file not found: {config_file}")
        sys.exit(1)

    config = OmegaConf.load(config_file)
    OmegaConf.resolve(config)

    for override_item in override:
        if "=" not in override_item:
            logging.error(f"Invalid override format: {override_item}")
            sys.exit(1)
        key, value = override_item.split("=", 1)
        try:
            import ast

            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value
        OmegaConf.update(config, key, parsed_value, merge=True)

        if key == "expt.seed":
            config = update_seed_in_config(config, parsed_value)

    expt_name = config.get("expt", {}).get("name", "default_expt")
    model_name = config.get("expt", {}).get("model", "default_model")
    outputs_dir = Path("outputs") / model_name / expt_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = outputs_dir / "resolved_config.yaml"
    OmegaConf.save(config, resolved_config_path)
    logging.info(f"Resolved config saved to {resolved_config_path}")

    cli_args = [mode, "--config", str(resolved_config_path)]
    sys.argv = [sys.argv[0]] + cli_args

    FinetuneCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


def update_seed_in_config(config, new_seed):
    if config.get("seed_everything"):
        config["seed_everything"] = new_seed

    if config.get("expt", {}).get("name"):
        config["expt"]["name"] = re.sub(
            r"s-(\d+)", f"s-{new_seed}", config["expt"]["name"]
        )

    if config.get("trainer", {}).get("default_root_dir"):
        config["trainer"]["default_root_dir"] = re.sub(
            r"s-(\d+)", f"s-{new_seed}", config["trainer"]["default_root_dir"]
        )

    if (
        config.get("trainer", {})
        .get("logger", ())
        .get("init_args", {})
        .get("save_dir", {})
    ):
        config["trainer"]["logger"]["init_args"]["save_dir"] = re.sub(
            r"s-(\d+)",
            f"s-{new_seed}",
            config["trainer"]["logger"]["init_args"]["save_dir"],
        )

    callbacks = config.get("trainer", {}).get("callbacks", {})
    if callbacks and len(callbacks) > 0:
        for callback in callbacks:
            if "ModelCheckpoint" in callback["class_path"] and callback.get(
                "init_args", {}
            ).get("dirpath"):
                callback["init_args"]["dirpath"] = re.sub(
                    r"s-(\d+)", f"s-{new_seed}", callback["init_args"]["dirpath"]
                )

    if config.get("ckpt_path"):
        new_path = re.sub(r"best-(\d+)", f"best-{new_seed}", config["ckpt_path"])
        if os.path.exists(new_path):
            config["ckpt_path"] = new_path
        else:
            config["ckpt_path"] = ""

    return config


def find_config_file(model, dataset_group, split_strategy, dataset, seed):
    config_filename = f"config-{seed}.yaml"
    config_path = (
        Path(CONFIGS_DIR)
        / model
        / dataset_group
        / split_strategy
        / dataset
        / config_filename
    )
    return str(config_path)


if __name__ == "__main__":
    main()
