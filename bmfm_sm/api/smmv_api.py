# ruff: noqa
import csv
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from bmfm_sm.api.utils import fix_finetuning_args
from bmfm_sm.api.dataset_registry import DatasetRegistry
from bmfm_sm.api.model_registry import ModelRegistry
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy, Modality, TaskType
from bmfm_sm.core.modules.base_pretrained_model import (
    BaseAndHeadModule,
    MultiTaskPredictionHead,
)
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import (
    Graph2dFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.image_finetune_dataset import (
    ImageFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.text_finetune_dataset import (
    TextFinetuneDataPipeline,
)
from bmfm_sm.predictive.modules.smmv_model import SmallMoleculeMultiView
from bmfm_sm.api.smmv_pretrained_model import (
    SmallMoleculeMultiViewPretrainedModel,
    SmallMoleculeMultiViewFinetunedModel,
)


class SmallMoleculeMultiViewModel(nn.Module):
    @staticmethod
    def from_pretrained(
        fusion_strategy, model_path=None, inference_mode=True, huggingface=False
    ):
        assert fusion_strategy in LateFusionStrategy

        if huggingface:
            if model_path:
                logging.info(
                    f"Loading checkpoint via HuggingFace Hub from provided path {model_path}"
                )
            elif model_path is None:
                model_path = ModelRegistry.get_checkpoint(Modality.MULTIVIEW)
                logging.info(
                    f"Loading checkpoint via HuggingFace Hub from default path {model_path}"
                )
            model = SmallMoleculeMultiViewPretrainedModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                agg_arch=fusion_strategy.agg_arch,
                agg_gate_input=fusion_strategy.projection_type,
                inference_mode=inference_mode,
            )
        else:
            model = SmallMoleculeMultiView(
                agg_arch=fusion_strategy.agg_arch,
                agg_gate_input=fusion_strategy.projection_type,
                inference_mode=inference_mode,
            )

            if model_path:
                logging.info(f"Loading checkpoint from provided path {model_path}")
                model.load_ckpt(model_path)
            elif model_path is None:
                checkpoint = ModelRegistry.get_checkpoint(Modality.MULTIVIEW)
                logging.info(f"Loading checkpoint from default path {checkpoint}")
                model.load_ckpt(checkpoint)
            elif model_path is False:
                logging.info("Not using checkpoint for model initialization")

        if inference_mode:
            model.eval()
        return model

    @staticmethod
    def from_finetuned(
        dataset,
        model_path=None,
        inference_mode=True,
        seed=101,
        split_strategy="ligand_scaffold",
        huggingface=False,
    ):
        if huggingface:
            if model_path:
                logging.info(
                    f"Loading checkpoint via HuggingFace Hub from provided path {model_path}"
                )
            elif model_path is None:
                if type(dataset) is str:
                    dataset = DatasetRegistry().get_dataset_info(dataset.upper())
                model_path = DatasetRegistry.get_checkpoint(
                    Modality.MULTIVIEW,
                    dataset,
                    seed=seed,
                    split_strategy=split_strategy,
                )
                logging.info(
                    f"Loading checkpoint via HuggingFace Hub from default path {model_path}"
                )
            if model_path.endswith(".ckpt"):
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                ckpt = torch.load(model_path, map_location=device)

                model_params = ckpt["hyper_parameters"]["model_params"]
                temp_model = SmallMoleculeMultiView(
                    agg_arch=model_params.get("agg_arch", "coeff_mlp"),
                    agg_gate_input=model_params.get("agg_gate_input", "projected"),
                    agg_weight_freeze=model_params.get("agg_weight_freeze", "unfrozen"),
                )

                hyperparams = fix_finetuning_args(
                    ckpt["hyper_parameters"]["finetuning_args"]
                )

                combined_model = SmallMoleculeMultiViewFinetunedModel.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    agg_arch=model_params.get("agg_arch", "coeff_mlp"),
                    agg_gate_input=model_params.get("agg_gate_input", "projected"),
                    agg_weight_freeze=model_params.get("agg_weight_freeze", "unfrozen"),
                    inference_mode=inference_mode,
                    input_dim=temp_model.get_embed_dim(),
                    num_tasks=ckpt["hyper_parameters"].get("num_tasks"),
                    task_type=ckpt["hyper_parameters"].get("task_type"),
                    head=hyperparams.get("head_arch", "mlp"),
                    hidden_dims=hyperparams.get("mlp_hidden_dims", [512, 384]),
                    use_norm=hyperparams.get("use_norm", True),
                    activation=hyperparams.get("head_activation", nn.GELU),
                    dropout_prob=hyperparams.get("head_dropout", 0.2),
                )
            else:
                combined_model = SmallMoleculeMultiViewFinetunedModel.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    inference_mode=inference_mode,
                )
        else:
            if model_path:
                checkpoint = model_path
            else:
                if type(dataset) is str:
                    dataset = DatasetRegistry().get_dataset_info(dataset.upper())
                checkpoint = DatasetRegistry.get_checkpoint(
                    Modality.MULTIVIEW,
                    dataset,
                    seed=seed,
                    split_strategy=split_strategy,
                )

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            ckpt = torch.load(checkpoint, map_location=device)

            model_params = ckpt["hyper_parameters"]["model_params"]
            model = SmallMoleculeMultiView(
                agg_arch=model_params.get("agg_arch", "coeff_mlp"),
                agg_gate_input=model_params.get("agg_gate_input", "projected"),
                agg_weight_freeze=model_params.get("agg_weight_freeze", "unfrozen"),
                inference_mode=inference_mode,
            )

            hyperparams = fix_finetuning_args(
                ckpt["hyper_parameters"]["finetuning_args"]
            )

            head = MultiTaskPredictionHead(
                input_dim=model.get_embed_dim(),
                num_tasks=ckpt["hyper_parameters"].get("num_tasks"),
                task_type=ckpt["hyper_parameters"].get("task_type"),
                head=hyperparams.get("head_arch", "mlp"),
                hidden_dims=hyperparams.get("mlp_hidden_dims", [512, 384]),
                use_norm=hyperparams.get("use_norm", True),
                activation=hyperparams.get("head_activation", nn.GELU),
                dropout_prob=hyperparams.get("head_dropout", 0.2),
            )

            model.load_ckpt(checkpoint)
            head.load_ckpt(checkpoint)
            combined_model = BaseAndHeadModule(model, head)
        if inference_mode:
            combined_model.eval()

        return combined_model

    @staticmethod
    def get_embeddings(
        smiles,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,
        model_path=None,
        pretrained_model=None,
        huggingface=False,
        **kwargs,
    ):
        if not pretrained_model:
            pretrained_model = SmallMoleculeMultiViewModel.from_pretrained(
                fusion_strategy,
                model_path=model_path,
                inference_mode=True,
                huggingface=huggingface,
            )
        joint_dict = {}
        joint_dict.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(smiles))
        joint_dict.update(TextFinetuneDataPipeline.smiles_to_text_format(smiles))
        joint_dict.update(ImageFinetuneDataPipeline.smiles_to_image_format(smiles))
        return pretrained_model.get_embeddings(joint_dict, **kwargs)

    @staticmethod
    def get_predictions(smile, dataset, model_path=None, finetuned_model=None):
        if not finetuned_model:
            finetuned_model = SmallMoleculeMultiViewModel.from_finetuned(
                dataset=dataset, model_path=model_path
            )
        joint_dict = {}
        joint_dict.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(smile))
        joint_dict.update(TextFinetuneDataPipeline.smiles_to_text_format(smile))
        joint_dict.update(ImageFinetuneDataPipeline.smiles_to_image_format(smile))
        predictions = finetuned_model(joint_dict).squeeze()

        if dataset.task_type == TaskType.CLASSIFICATION:
            sigmoid_output = torch.sigmoid(predictions)
            binary_output = (sigmoid_output > 0.5).int()
            return binary_output
        else:
            return predictions


class PredictionIterator:
    def __init__(self, dataset, model_path=None):
        self.dataset = dataset
        self.finetuned_model = SmallMoleculeMultiViewModel.from_finetuned(
            dataset=self.dataset, model_path=model_path, inference_mode=True
        )

        self.file_path = os.path.join(os.environ["BMFM_HOME"], dataset.path)
        self.file = open(self.file_path)
        self.csv_reader = csv.reader(self.file)
        next(self.csv_reader, None)  # Skip the header

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.csv_reader, None)
        if row is not None:
            smiles, labels = row[1], row[2].split()
            preds = SmallMoleculeMultiViewModel.get_predictions(
                smiles, dataset=self.dataset, finetuned_model=self.finetuned_model
            ).numpy()

            if self.dataset.task_type == TaskType.CLASSIFICATION:
                return smiles, np.asarray(labels, dtype=int), preds
            else:
                return smiles, np.asarray(labels, dtype=float), preds
        else:
            self.file.close()
            raise StopIteration
