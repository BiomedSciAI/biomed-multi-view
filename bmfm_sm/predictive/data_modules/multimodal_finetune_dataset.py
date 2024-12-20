import logging

import torch
from fuse.data.pipelines.pipeline_default import PipelineDefault

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import (
    Graph2dFinetuneDataPipeline,
    Graph2dGNNFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.image_finetune_dataset import (
    ImageFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.mpp_finetune_dataset import MPPFinetuneDataset
from bmfm_sm.predictive.data_modules.text_finetune_dataset import (
    TextFinetuneDataPipeline,
)


class MultiModalFinetuneDataPipeline(MPPFinetuneDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        self.task_type = dataset_args.get("task_type")
        logging.info(f"Working on task {self.task_type }")

        # Checking which modalities are a part of this MultiModalDataset
        self.TEXT_MODEL = "TEXT_MODEL" in dataset_args["modalities"]
        self.IMAGE_MODEL = "IMAGE_MODEL" in dataset_args["modalities"]
        self.GRAPH_2D_MODEL = "GRAPH_2D_MODEL" in dataset_args["modalities"]
        self.GRAPH_GNN_MODEL = (
            "GRAPH_GCN_MODEL" in dataset_args["modalities"]
            or "GRAPH_GIN_MODEL" in dataset_args["modalities"]
            or "GRAPH_ATTENTIVEFP_MODEL" in dataset_args["modalities"]
            or "GRAPH_TRIMNET_MODEL" in dataset_args["modalities"]
        )

        modalities = ",".join(dataset_args["modalities"])
        logging.info(f"modalities running in data pipeline {modalities}")

        # Initializing some variables needed by specific modalities
        if self.TEXT_MODEL:
            self.text_pipeline = TextFinetuneDataPipeline(
                data_dir=data_dir, dataset_args=dataset_args, stage=stage
            )

        if self.IMAGE_MODEL:
            self.image_pipeline = ImageFinetuneDataPipeline(
                data_dir=data_dir, dataset_args=dataset_args, stage=stage
            )

        if self.GRAPH_2D_MODEL:
            self.graph2d_pipeline = Graph2dFinetuneDataPipeline(
                data_dir=data_dir, dataset_args=dataset_args, stage=stage
            )

        if self.GRAPH_GNN_MODEL:
            self.graph2d_gnn_pipeline = Graph2dGNNFinetuneDataPipeline(
                data_dir=data_dir, dataset_args=dataset_args, stage=stage
            )

        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        # Get the ops from the superclass (Formatting the MPP Labels)
        all_ops = super().get_ops_pipeline()._ops_and_kwargs

        # Get the ops from each of the included modality pipelines (Doing [1:] to remove repeats of the OpFormatMPP)
        if self.TEXT_MODEL:
            all_ops.extend(self.text_pipeline.get_ops_pipeline()._ops_and_kwargs[1:])

        if self.IMAGE_MODEL:
            all_ops.extend(self.image_pipeline.get_ops_pipeline()._ops_and_kwargs[1:])

        if self.GRAPH_2D_MODEL:
            all_ops.extend(self.graph2d_pipeline.get_ops_pipeline()._ops_and_kwargs[1:])

        if self.GRAPH_GNN_MODEL:
            all_ops.extend(
                self.graph2d_gnn_pipeline.get_ops_pipeline()._ops_and_kwargs[1:]
            )

        return PipelineDefault("all_ops", all_ops)

    def collate_fn(self, batch):
        collated = {}

        if self.IMAGE_MODEL:
            images, labels = self.image_pipeline.collate_fn(batch)
            collated[ns.FIELD_IMAGE] = images
            collated[ns.FIELD_LABEL] = labels

        if self.TEXT_MODEL:
            tokenized_smiles, attention_mask, labels = self.text_pipeline.collate_fn(
                batch
            )
            collated[ns.FIELD_TOKENIZED_SMILES] = tokenized_smiles
            collated["attention_mask"] = attention_mask
            if ns.FIELD_LABEL not in collated:
                # Same as the Image Labels so ok to overwrite
                collated[ns.FIELD_LABEL] = labels

        if self.GRAPH_2D_MODEL:
            graph_2d_collated = self.graph2d_pipeline.collate_fn(batch)
            collated.update(graph_2d_collated)
            if ns.FIELD_LABEL not in collated:
                collated[ns.FIELD_LABEL] = graph_2d_collated["y"]
            else:
                del collated["y"]

        if self.GRAPH_GNN_MODEL:
            graph_gnn_batch_data = self.graph2d_gnn_pipeline.collate_fn(batch)
            collated["bd"] = graph_gnn_batch_data
            if ns.FIELD_LABEL not in collated:
                collated[ns.FIELD_LABEL] = graph_gnn_batch_data["y"]

        collated[ns.FIELD_LABEL] = collated[ns.FIELD_LABEL].to(torch.float32)
        
        return collated

    def get_feature_fn(self):
        def f(example):
            if "graph2d" in example:
                return example["graph2d"].x.shape[0]  # num_atoms
            return len(example["smiles"])

        return f
