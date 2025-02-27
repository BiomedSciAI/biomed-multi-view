import os
import logging
import sys
from bmfm_sm.core.data_modules.splits import create_splits, calculate_task_weights
import shutil
import json
from importlib import resources
import yaml
from omegaconf import OmegaConf

BMFM_HOME = os.environ.get("BMFM_HOME")
if not BMFM_HOME:
    logging.error("BMFM_HOME environment variable is not set.")
    sys.exit(1)
logging.info(f"BMFM_HOME set to {BMFM_HOME}")

def add_new_dataset_group(
    dataset_files,
    dataset_group = 'other',
    split_type = 'ligand_scaffold',
    include_class_weights = False,
    
):
    """
        args:
        - dataset_files --> Dictionary mapping dataset names to CSV paths
        - dataset_group --> Name of the datasetg group for all the individual datasets
        - split_type --> Name of split_type (e.g. random, ligand_scaffold, ligand_random_scaffold, etc.)

    """
    
    # Make the necessary output directories in the BMFM_HOME root
    os.makedirs(os.path.join(BMFM_HOME, f"datasets/raw_data/{dataset_group}"), exist_ok=True)
    os.makedirs(os.path.join(BMFM_HOME, f"datasets/splits/{dataset_group}/{split_type}"), exist_ok=True)
    os.makedirs(os.path.join(BMFM_HOME, f"configs_finetune/MULTIVIEW_MODEL/{dataset_group}/{split_type}"), exist_ok=True)
    
    frac_val, frac_test = 0.1, 0.1
    frac_train = 0.8 if split_type == "ligand_scaffold_balanced" else 1.0
    
    for file_pair in dataset_files.items():
        dataset, file_path = file_pair
        
        # Copy over the raw data file
        new_file_path = os.path.join(BMFM_HOME, f"datasets/raw_data/{dataset_group}/{dataset}.csv")
        shutil.copy(file_path, new_file_path)
        print(f"Raw CSV for dataset {dataset} copied to data root succesfully.")
        
        # Create the splits and store them in the correct location
        dict_split = create_splits(
                    source_file=new_file_path,
                    frac_train=frac_train,
                    frac_val=frac_val,
                    frac_test=frac_test,
                    strategy=split_type,
                )
        new_split_file = os.path.join(BMFM_HOME, f"datasets/splits/{dataset_group}/{split_type}", dataset + "_split.json")
        with open(new_split_file, "w") as json_file:
            json.dump(dict_split, json_file)
        print(f"Split JSON for dataset {dataset} and split_type {split_type} created succesfully.")
        
        # Create a config file for this dataset and split type, and store in the root
        new_config_dir = os.path.join(BMFM_HOME, f"configs_finetune/MULTIVIEW_MODEL/{dataset_group}/{split_type}/{dataset}")
        os.makedirs(new_config_dir, exist_ok=True)
        
        with resources.open_text("bmfm_sm.resources", "finetune_config_template.yaml") as f:
            config = yaml.safe_load(f)
            
        config = OmegaConf.create(config)
        config["expt"]["dataset"] = dataset
        config["expt"]["split_strategy"] = split_type
        config["data"]["init_args"]["dataset_args"]["split_file"] = new_split_file
        config["data"]["init_args"]["data_dir"] = os.path.join(BMFM_HOME, f"datasets/raw_data/{dataset_group}")
        
        if include_class_weights:
            weights = calculate_task_weights(dict_split['train'], new_file_path)
            config["model"]["init_args"]["finetuning_args"]["class_weights"] = weights.tolist()
            
        OmegaConf.resolve(config)
        OmegaConf.save(config, os.path.join(new_config_dir, f"config-{config['expt']['seed']}.yaml"))
        print(f"Config for dataset {dataset} created succesfully. \n")
        