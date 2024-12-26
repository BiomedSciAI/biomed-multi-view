# DISCLAIMER: PLEASE DO NOT CHANGE THIS FILE WITHOUT EXPLICITLY CONSULTING THE BMFM-SM TEAM. ANY CHANGES MADE HERE WILL IMPACT ALL THE PRETRAINED MODELS
"""
All the pretrained models (unimodal and multi-modal) should inherit from this parent class. Each model will declare its modality (using the modality class above) and
be required to implement the load_ckpt and get_embeddings methods, along with typical nn.Module methods (forward, etc.).
"""


import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bmfm_sm.core.data_modules.namespace import Modality, TaskType


class BaseModel(nn.Module):
    def __init__(self, modality, **kwargs):
        super().__init__(**kwargs)

        # Modality can be passed in as a string ("image"), an integer (1) or an enum value (Modality.IMAGE)
        if isinstance(modality, str):
            self.model_modality = Modality[modality.upper()]
        elif isinstance(modality, int):
            self.model_modality = Modality(modality)
        elif isinstance(modality, Modality):
            self.model_modality = modality
        else:
            raise ValueError("Invalid modality argument.")

    @property
    def modality(self):
        return self.model_modality

    # Load's in the checkpoint and updates the model's state dictionary with that of the checkpoint. Can be overriden by children classes if they have to
    # load checkpoint in a specific way
    def load_ckpt(self, path_to_ckpt):
        if torch.cuda.is_available():
            checkpoint = torch.load(path_to_ckpt)
        else:
            checkpoint = torch.load(path_to_ckpt, map_location=torch.device("cpu"))

        ckpt_keys = list(checkpoint["state_dict"].keys())
        model_keys = list(self.state_dict().keys())
        model_sd = self.state_dict()

        logging.info(
            f"Length of checkpoint keys {len(ckpt_keys)} and length of model keys {len(model_keys)}"
        )
        for ckpt_key, model_key in zip(ckpt_keys, model_keys):
            model_sd[model_key] = checkpoint["state_dict"][ckpt_key]

        self.load_state_dict(model_sd)

        logging.info("loaded ckpt: %s" % path_to_ckpt)

    @abstractmethod
    # Should return the model embeddings for the input x. Will eventually be used to get modality-specific or multi-modality embeddings for input
    def get_embeddings(self, x):
        pass

    @abstractmethod
    def get_embed_dim(self):
        pass

    @abstractmethod
    # Takes in a SMILES, preprocesses it as necessary for the model, loads the finetuned model checkpoint for that dataset (or from test_ckpt if specified) and gives a prediction
    # For a classification task, apply a sigmoid or similar before to make sure it's in the correct space
    def get_predictions(smiles, dataset, test_ckpt=None):
        pass


# Utility module for stitching together a base finetuned model with its prediction head
class BaseAndHeadModule(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        # Initialize Module1 and Module2 as submodules
        self.model = model
        self.head = head

        # Disable gradient tracking for model and head parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(False)

    def forward(self, *args):
        with torch.no_grad():
            return self.head(self.model(*args))

    # Use Case: User doesn't want prediction from the head, they want the embeddings from the base model
    def get_embeddings(self, x):
        return self.model.get_embeddings(x)


# BASE PREDICTION, CLASSIFICATION AND REGRESSION HEADS
# - All prediction heads implemented should inherit from the BasePredictionHead and implement the __init__, forward and compute_loss methods
# - Users can make use of the classification/multi-classification/regression/multi-regression heads within their defined prediction heads to avoid code repetion. They will still
#     need to add a forward and compute_loss method


class BasePredictionHead(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, sequence_output, pooled_output):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        raise NotImplementedError


class MultiTaskPredictionHead(nn.Module):
    activations = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}

    def __init__(
        self,
        input_dim,
        num_tasks,
        task_type: TaskType,
        num_classes_per_task=1,
        head="mlp",
        hidden_dims=[512, 512],
        activation=nn.GELU,
        use_norm=False,
        dropout_prob=0.2,
        softmax=False,
        bias_terms=[],
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.task_type = task_type
        self.softmax = softmax

        if activation is None:
            self.activation = None
        elif type(activation) is str:
            self.activation = MultiTaskPredictionHead.activations[activation.lower()]()
        else:
            self.activation = activation()

        if task_type == TaskType.REGRESSION:
            assert num_classes_per_task == 1

        self.shared_task_head = self._build_head(
            input_dim, hidden_dims, activation, use_norm, dropout_prob, head, bias_terms
        )

        if isinstance(self.num_classes_per_task, list):
            self.output_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_dims[-1], num_classes)
                    for num_classes in self.num_classes_per_task
                ]
            )

    def forward(self, x):
        task_logits = self.shared_task_head(x)

        if isinstance(self.num_classes_per_task, int):
            task_logits = task_logits.view(
                -1, self.num_tasks, self.num_classes_per_task
            )
        else:
            task_logits = [
                output_layer(task_logits) for output_layer in self.output_layers
            ]

        if self.task_type == TaskType.CLASSIFICATION and self.softmax:
            if isinstance(self.num_classes_per_task, int):
                return F.softmax(task_logits, dim=2)
            else:
                return [F.softmax(logit, dim=-1) for logit in task_logits]
        else:
            return task_logits

    def _build_head(
        self,
        input_dim,
        hidden_dims,
        activation,
        use_norm,
        dropout_prob,
        head,
        bias_terms,
    ):
        layers = []
        layer_num = -1
        if head == "mlp":
            layers.append(nn.Linear(input_dim, hidden_dims[0]))

            if use_norm:
                layers.append(nn.LayerNorm(hidden_dims[0]))
            if activation:
                layers.append(self.activation)
            if dropout_prob != 0:
                layers.append(nn.Dropout(dropout_prob))

            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                if use_norm:
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                if activation:
                    layers.append(self.activation)
                if (i < len(hidden_dims) - 1) and dropout_prob != 0:
                    layers.append(nn.Dropout(dropout_prob))

            if isinstance(self.num_classes_per_task, int):
                layers.append(
                    nn.Linear(
                        hidden_dims[-1], self.num_tasks * self.num_classes_per_task
                    )
                )

        else:  # Linear case
            layers.append(
                nn.Linear(input_dim, self.num_tasks * self.num_classes_per_task)
            )
            if use_norm:
                layers.append(nn.LayerNorm(self.num_tasks * self.num_classes_per_task))
                layer_num = layer_num - 1
            if activation is not None:
                layers.append(self.activation)
                layer_num = layer_num - 1
        bias_terms = np.array(list(bias_terms))

        if bias_terms.size != 0:
            assert len(bias_terms) == len(
                layers[layer_num].bias.detach().tolist()
            ), f"Length of bias_values must match output dimension: {len(bias_terms)}, {len(layers[layer_num].bias.detach().tolist())}"

            with torch.no_grad():
                layers[layer_num].bias.data = torch.tensor(
                    bias_terms, dtype=layers[layer_num].bias.dtype
                ).view(-1)

        return nn.Sequential(*layers)

    # Special method for this prediction head since we will be restoring it when serving finetuned models

    def load_ckpt(self, path_to_ckpt):
        if torch.cuda.is_available():
            ckpt = torch.load(path_to_ckpt)
        else:
            ckpt = torch.load(path_to_ckpt, map_location=torch.device("cpu"))

        # Accounting for checkpoints that are saved in the Lora SD pattern
        if "lora_head" in ckpt["state_dict"].keys():
            self.load_state_dict(ckpt["state_dict"]["lora_head"])
            return

        ckpt_keys = [key for key in ckpt["state_dict"].keys() if "pred_head" in key]
        head_keys = list(self.state_dict())
        head_sd = self.state_dict()

        assert len(head_keys) == len(ckpt_keys)

        for ckpt_key, head_key in zip(ckpt_keys, head_keys):
            head_sd[head_key] = ckpt["state_dict"][ckpt_key]

        self.load_state_dict(head_sd)

        logging.info(f"Loading finetune checkpoint for Prediction Head: {path_to_ckpt}")


class Parallel(nn.ModuleList):
    """Runs modules in parallel on the same input and merges their results element wise."""

    def __init__(self, *modules: nn.Module, merge: str | Callable = "sum"):
        """
        Runs modules in parallel on the some input and merges their results element wise.

        Args:
        ----
            merge: operation for merging list of results (default: `"sum"`)

        """
        super().__init__(modules)
        self.merge = create_merge(merge)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge([module(x) for module in self])


MERGE_METHODS: dict[str, Callable] = {
    "cat": lambda xs: torch.cat(xs, dim=1),
    "sum": lambda xs: sum(xs),  # type: ignore
    "prod": lambda xs: reduce(lambda x, y: x * y, xs),  # type: ignore
}


def create_merge(merge: str | Callable) -> Callable:
    return MERGE_METHODS[merge] if isinstance(merge, str) else merge


class AttentionalViewAggregator(nn.Module):
    def __init__(
        self,
        dim_list,
        path_to_ckpt=None,
        arch_type="concat",
        hidden_dims=[512, 512, 512],
        activation="ReLU",
        use_norm=True,
        dropout_prob=0.1,
        gate_input="unprojected",
    ):
        super().__init__()
        self.arch_type = arch_type
        logging.info(f"Using {arch_type} architecture for aggregator")
        logging.info(f"dim_list {dim_list} for aggregator")

        if self.arch_type == "concat":
            self.output_dim = sum(dim_list)
        else:
            # Projecting down to the lowest dim for the models (512); Not needed by concat architecture
            min_dim_list = min(dim_list)
            self.projections = nn.ModuleList(
                [nn.Linear(dim, min_dim_list) for dim in dim_list]
            )

        if (
            self.arch_type == "moe_weighted_concat"
            or self.arch_type == "moe_noised_weighted_concat"
        ):
            network_class = (
                WeightingNetwork
                if self.arch_type == "moe_weighted_concat"
                else NoisedWeightingNetwork
            )
            if gate_input == "projected":
                gating_network = network_class(
                    min(dim_list) * len(dim_list), num_experts=len(dim_list)
                )
                self.weighted_concat_network = WeightedConcatenationMoE(
                    gating_network, dim_list, gate_input="projected"
                )
                self.output_dim = min(dim_list) * len(dim_list)
            elif gate_input == "unprojected":
                gating_network = network_class(sum(dim_list), num_experts=len(dim_list))
                self.weighted_concat_network = WeightedConcatenationMoE(
                    gating_network, dim_list, gate_input="unprojected"
                )
                self.output_dim = sum(dim_list)
            elif gate_input == "both_projected":
                gating_network = network_class(
                    min(dim_list) * len(dim_list), num_experts=len(dim_list)
                )
                self.weighted_concat_network = WeightedConcatenationMoE(
                    gating_network, dim_list, gate_input="both_projected"
                )
                self.output_dim = min(dim_list)
            else:
                raise ValueError("gate_input has to be projected or unprojected")

        if self.arch_type == "coeff" or self.arch_type == "coeff_mlp":
            self.w_before_mean = nn.Sequential(
                nn.Linear(min_dim_list, min_dim_list),
                nn.Tanh(),
                nn.Linear(min_dim_list, 1, bias=False),
            )
            self.w_before_mean.apply(self.init_weights)
            self.down_project = nn.Linear(min_dim_list * len(dim_list), min_dim_list)
            self.output_dim = min_dim_list
            if self.arch_type == "coeff_mlp":
                layers = []
                layers.append(nn.Linear(min_dim_list, hidden_dims[0]))
                if use_norm:
                    layers.append(nn.LayerNorm(hidden_dims[0]))
                layers.append(self.activation_fn(activation))
                layers.append(nn.Dropout(dropout_prob))
                for i in range(1, len(hidden_dims)):
                    layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                    if use_norm:
                        layers.append(nn.LayerNorm(hidden_dims[i]))
                    layers.append(self.activation_fn(activation))
                    layers.append(nn.Dropout(dropout_prob))
                self.shared_task_head = nn.Sequential(*layers)

        if path_to_ckpt is not None:
            self.load_ckpt(path_to_ckpt)

    def activation_fn(self, activation: str = "ReLU"):
        return getattr(torch.nn, activation)()

    def forward(self, outputs):
        if self.arch_type == "concat":
            return torch.cat(outputs, dim=1), None

        projection_outputs = [
            proj(output) for proj, output in zip(self.projections, outputs)
        ]
        combined_output = torch.stack(projection_outputs, dim=1)

        if (
            self.arch_type == "moe_weighted_concat"
            or self.arch_type == "moe_noised_weighted_concat"
        ):
            if (
                self.weighted_concat_network.gate_input == "projected"
                or self.weighted_concat_network.gate_input == "both_projected"
            ):
                output, avg_weights = self.weighted_concat_network(combined_output)
            elif self.weighted_concat_network.gate_input == "unprojected":
                output, avg_weights = self.weighted_concat_network(
                    torch.cat(outputs, dim=1)
                )
            return output, avg_weights

        tmp = torch.nn.functional.normalize(combined_output, dim=-1)
        w = self.w_before_mean(tmp).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((combined_output.shape[0],) + beta.shape)
        logits = beta * combined_output
        coeffs = beta.squeeze(2)[0]
        flat_logits = torch.flatten(logits, start_dim=1)
        down_logits = self.down_project(flat_logits)
        if self.arch_type == "coeff_mlp":
            down_logits = self.shared_task_head(down_logits)
        return down_logits, coeffs

    def load_ckpt(self, path_to_ckpt, prefix="aggregator."):
        if torch.cuda.is_available():
            checkpoint = torch.load(path_to_ckpt)
        else:
            checkpoint = torch.load(path_to_ckpt, map_location=torch.device("cpu"))

        ckpt_keys = list(checkpoint["state_dict"].keys())
        model_keys = list(self.state_dict().keys())
        model_sd = self.state_dict()

        logging.info(
            f"AttentionalViewAggregator: Length of checkpoint keys {len(ckpt_keys)} and length of model keys {len(model_keys)}"
        )

        for model_key in model_keys:
            model_sd[model_key] = checkpoint["state_dict"][prefix + model_key]

        self.load_state_dict(model_sd)

        logging.info("AttentionalViewAggregator: loaded ckpt: %s" % path_to_ckpt)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        return

    def get_output_dim(self):
        return self.output_dim


class WeightingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        weights = torch.softmax(self.fc(x), dim=-1)
        return weights


class NoisedWeightingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, noise_level=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.noise_level = noise_level

    def forward(self, x):
        weights = self.fc(x)
        noise = torch.randn_like(weights) * self.noise_level
        return torch.softmax(weights + noise, dim=-1)


class WeightedConcatenationMoE(nn.Module):
    def __init__(self, gating_network, dim_list, gate_input="projected"):
        super().__init__()
        self.gating_network = gating_network
        self.dim_list = dim_list
        self.gate_input = gate_input

    def forward(self, combined_outputs):
        if self.gate_input == "projected":
            batch_size, _, _ = combined_outputs.shape
            concat_embeddings = combined_outputs.reshape(batch_size, -1)
            weights = self.gating_network(concat_embeddings)
            expert_outputs = list(torch.unbind(combined_outputs, dim=1))
        elif self.gate_input == "unprojected":
            weights = self.gating_network(combined_outputs)
            expert_outputs = torch.split(combined_outputs, self.dim_list, dim=1)
        elif self.gate_input == "both_projected":
            batch_size, _, _ = combined_outputs.shape
            concat_embeddings = combined_outputs.reshape(batch_size, -1)
            weights = self.gating_network(concat_embeddings)
            expert_outputs = list(torch.unbind(combined_outputs, dim=1))
            weighted_expert_outputs = [
                weights[:, i].unsqueeze(1) * output
                for i, output in enumerate(expert_outputs)
            ]
            summed_output = torch.stack(weighted_expert_outputs).sum(dim=0)
            avg_weights_for_batch = torch.sum(weights, dim=0).float() / weights.size(0)
            return summed_output, avg_weights_for_batch
        else:
            raise ValueError("gate_input has to be either projected or unprojected")

        weighted_expert_outputs = [
            weights[:, i].unsqueeze(1) * output
            for i, output in enumerate(expert_outputs)
        ]
        concatenated_output = torch.cat(weighted_expert_outputs, dim=-1)
        avg_weights_for_batch = torch.sum(weights, dim=0).float() / weights.size(0)
        return concatenated_output, avg_weights_for_batch
