from importlib import resources
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from bmfm_sm.api.utils import fix_finetuning_args
from bmfm_sm.core.data_modules.namespace import TaskType
from bmfm_sm.core.modules.base_pretrained_model import MultiTaskPredictionHead
from bmfm_sm.predictive.modules.smmv_model import SmallMoleculeMultiView

with resources.open_text("bmfm_sm.resources", "modelcard_template.md") as f:
    model_card_template = f.read()


library_name = "SmallMoleculeMultiView"
tags = [
    "drug-discovery",
    "small-molecules",
    "multimodal",
    "virtual-screening",
    "molecules",
    "multi-view",
    "chemistry",
    "bio-medical",
    "molecular-property-prediction",
    "moleculenet",
    "drug-target-interaction",
    "binding-affinity-prediction",
]
repo_url = "https://github.com/BiomedSciAI/biomed-multi-view"
license = "apache-2.0"


class SmallMoleculeMultiViewPretrainedModel(
    SmallMoleculeMultiView,
    PyTorchModelHubMixin,
    library_name=library_name,
    tags=tags,
    repo_url=repo_url,
    model_card_template=model_card_template,
    license=license,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs,
    ):
        if pretrained_model_name_or_path.endswith(".pth"):
            model = cls(**model_kwargs)
            model.load_ckpt(pretrained_model_name_or_path)
        else:
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **model_kwargs,
            )
        return model


class SmallMoleculeMultiViewFinetunedModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name=library_name,
    tags=tags,
    repo_url=repo_url,
    model_card_template=model_card_template,
    license=license,
):
    def __init__(
        self,
        input_dim,
        num_tasks,
        task_type: TaskType,
        *args,
        agg_arch="concat",
        agg_gate_input="unprojected",
        agg_weight_freeze="unfrozen",
        inference_mode=False,
        num_classes_per_task=1,
        head="mlp",
        hidden_dims=[512, 512],
        activation=nn.GELU,
        use_norm=False,
        dropout_prob=0.2,
        softmax=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize Module1 and Module2 as submodules
        self.model = SmallMoleculeMultiView(
            agg_arch=agg_arch,
            agg_gate_input=agg_gate_input,
            agg_weight_freeze=agg_weight_freeze,
            inference_mode=inference_mode,
        )
        self.head = MultiTaskPredictionHead(
            input_dim=input_dim,
            num_tasks=num_tasks,
            task_type=task_type,
            num_classes_per_task=num_classes_per_task,
            head=head,
            hidden_dims=hidden_dims,
            use_norm=use_norm,
            activation=activation,
            dropout_prob=dropout_prob,
            softmax=softmax,
        )

        # Disable gradient tracking for model and head parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs,
    ):
        if pretrained_model_name_or_path.endswith(".ckpt"):
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            ckpt = torch.load(pretrained_model_name_or_path, map_location=device)
            hpars = ckpt["hyper_parameters"]

            model_params = ckpt["hyper_parameters"]["model_params"]
            temp_model = SmallMoleculeMultiView(
                agg_arch=model_params.get("agg_arch", "coeff_mlp"),
                agg_gate_input=model_params.get("agg_gate_input", "projected"),
                agg_weight_freeze=model_params.get("agg_weight_freeze", "unfrozen"),
            )

            model_kwargs["input_dim"] = temp_model.get_embed_dim()
            if "num_tasks" in hpars:
                model_kwargs["num_tasks"] = hpars.get("num_tasks")
            if "task_type" in hpars:
                model_kwargs["task_type"] = hpars.get("task_type")

            finetuning_args = fix_finetuning_args(hpars["finetuning_args"])
            model_kwargs["head"] = finetuning_args.get("head_arch", "mlp")
            model_kwargs["hidden_dims"] = finetuning_args.get(
                "mlp_hidden_dims", [512, 384]
            )
            model_kwargs["use_norm"] = finetuning_args.get("use_norm", True)
            model_kwargs["activation"] = finetuning_args.get("head_activation", nn.GELU)
            model_kwargs["dropout_prob"] = finetuning_args.get("head_dropout", 0.2)

            model_kwargs["agg_arch"] = model_params.get("agg_arch", "coeff_mlp")
            model_kwargs["agg_gate_input"] = model_params.get(
                "agg_gate_input", "projected"
            )
            model_kwargs["agg_weight_freeze"] = model_params.get(
                "agg_weight_freeze", "unfrozen"
            )
            combined_model = cls(**model_kwargs)
            combined_model.model.load_ckpt(pretrained_model_name_or_path)
            combined_model.head.load_ckpt(pretrained_model_name_or_path)
        else:
            combined_model = super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **model_kwargs,
            )
        return combined_model

    def forward(self, *args):
        with torch.no_grad():
            return self.head(self.model(*args))

    # Use Case: User doesn't want prediction from the head, they want the embeddings from the base model
    def get_embeddings(self, x):
        return self.model.get_embeddings(x)
