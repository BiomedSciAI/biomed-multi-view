import os
import re
from dataclasses import dataclass
from enum import Enum
from importlib import resources

import yaml

from bmfm_sm.core.data_modules.namespace import Metrics, Modality, TaskType


class DatasetRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.datasets = {}
            self.collections = {}
            with resources.open_text("bmfm_sm.resources", "dataset_registry.yaml") as f:
                dataset_info = yaml.safe_load(f)

            for item in dataset_info:
                metric = Metrics[dataset_info[item]["preferred_metric"].upper()]
                task_type = TaskType[dataset_info[item]["task_type"].upper()]
                num_tasks = dataset_info[item]["num_tasks"]
                desc = dataset_info[item]["description"]
                path = dataset_info[item]["path"]
                example = dataset_info[item]["example"]
                num_classes = (
                    dataset_info[item]["num_classes"]
                    if "num_classes" in dataset_info[item]
                    else None
                )
                collection = DatasetCollection(
                    dataset_info[item].get("collection", DatasetCollection.ALL.value)
                )
                ds = Dataset(
                    item,
                    num_tasks,
                    task_type,
                    desc,
                    metric,
                    path,
                    example,
                    collection,
                    num_classes,
                )
                self.datasets[item] = ds
                self.collections.setdefault(collection.value, []).append(ds)
                if collection.value != DatasetCollection.ALL.value:
                    self.collections.setdefault(DatasetCollection.ALL.value, []).append(
                        ds
                    )

            self.__initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def get_checkpoint(
        modality: Modality, dataset, seed=101, split_strategy="ligand_scaffold"
    ):
        # dataset must be of type Dataset (See bottom of this file)

        with resources.open_text("bmfm_sm.resources", "finetuned_ckpts.yaml") as f:
            config = yaml.safe_load(f)

            modality_checkpoints = config.get(modality.name, {})
            ckpt_relative_path = modality_checkpoints.get(dataset.dataset_name, None)

            if ckpt_relative_path:
                base_path = os.environ.get("BMFM_HOME")
                if base_path is None:
                    raise ValueError(
                        "Remember to set the BMFM_HOME environment variable"
                    )

                # If there is only 1 available split_strategy (e.g. ligand_balanced_scaffold), return the checkpoint associated with that
                # Else, provide the checkpoint for the specified split_strategy
                if len(ckpt_relative_path.keys()) == 1:
                    ckpt_relative_path = ckpt_relative_path[
                        next(iter(ckpt_relative_path))
                    ]
                else:
                    ckpt_relative_path = ckpt_relative_path[split_strategy]

                # Can specify and ask for a different seed than the default (101)
                if seed and re.search(r"-\d{3}\.ckpt$", ckpt_relative_path):
                    new_seed_str = f"{seed:03}"
                    ckpt_relative_path = re.sub(
                        r"-(\d{3})\.ckpt$", f"-{new_seed_str}.ckpt", ckpt_relative_path
                    )

                full_path = os.path.join(
                    base_path, "bmfm_model_dir/finetuned/", ckpt_relative_path
                )

                if not os.path.exists(full_path):
                    raise ValueError(
                        f"Checkpoint not available for modality {modality} and dataset {dataset.dataset_name} and seed {seed}"
                    )
                return full_path
            else:
                raise ValueError(
                    f"Checkpoint not available for modality {modality} and dataset {dataset.dataset_name}"
                )

    @classmethod
    def list_datasets(self):
        return list(self._instance.datasets.keys())

    @classmethod
    def list_collections(self):
        return list(self._instance.collections.keys())

    @classmethod
    def get_collection(self, collection: str):
        return self._instance.collections[collection]

    @classmethod
    def get_dataset_info(self, name: str):
        return self._instance.datasets.get(name, "Unknown dataset")


class DatasetCollection(Enum):
    MOLECULENET = "MoleculeNet"
    CYP = "CYP"
    ALL = "All"
    GPCR = "GPCR"
    COMPUTATIONALADME = "ComputationalADME"


@dataclass
# Dataset object specifying the name of the dataset, the associated number of tasks, type of tasks, a description and the preferred evaluation metric
class Dataset:
    dataset_name: str
    num_tasks: int
    task_type: TaskType
    description: str
    preferred_metric: Metrics
    path: str
    example: str
    collection: DatasetCollection
    num_classes: int = None  # Only needed for classification tasks

    def get_example_smiles(self):
        return self.example.split(",")[0]

    def get_example_labels(self):
        return self.example.split(",")[1]
