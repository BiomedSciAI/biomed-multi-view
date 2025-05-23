import importlib
import logging
import os

import loralib as lora
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pytorch_lightning.utilities import grad_norm

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.api.utils import fix_finetuning_args
from bmfm_sm.core.data_modules.namespace import TaskType
from bmfm_sm.core.modules.base_pretrained_model import Modality, MultiTaskPredictionHead

# Maps a modality to the corresponding location of its labels in the batch
modality_map = {
    Modality.IMAGE: 1,
    Modality.TEXT: 2,
    Modality.GRAPH: "y",
    Modality.GRAPH_3D: "y",
    Modality.MULTIVIEW: "label",
}


class FineTuneLightningModule(pl.LightningModule):
    def __init__(
        self,
        base_model_class,
        model_params,
        task_type="classification",
        num_tasks=617,
        checkpoint_path=None,
        lr=0.00001,
        weight_decay=0.01,
        finetuning_args={},
    ):
        """
        params:
            - base_model_class --> Pass in a BMFMPretrainedModel type class or a string path to the class (E.g. "bmfm_sm.predictive.modules.image_models.ImageModelForPretraining")
            - model_params --> Any parameters needed to initialize the base model (pass as a dictionary)
            - task_type --> ["classification", "regression"]
            - num_tasks --> Number of finetuning tasks (will be used for the output dimensionality of the prediction head)
            - checkpoint_path --> Path to the checkpoint for the base model; If none, then the base model will be randomly initialized
            - lr --> learning rate for finetuning
            - weight_decay --> weight decay used by AdamW optimizer
            - finetuning_args: The defaults are ("gradual", "default", no scheduler, "mlp", True)
                - weight_freeze --> Choose from ["frozen", "unfrozen", "gradual", "lora"].
                        For frozen, base model weights are kept frozen throughout rain.
                        For unfrozen, they are kept unfrozen the whole time.
                        For gradual, they will start off as frozen and be gradually unfrozen.
                        For lora, the LoRA scheme will be used to freeze original weights and rank-decomposition matrices will be learned instead
                - initialization --> Choose from ["default", "xavier", "he"]
                - scheduler --> Choose from ["cosine_anneal", "one_cycle", "poly", "step"]
                - head_arch --> Choose from ["linear", "mlp"]; Causes the prediction head to either have a simple linear layer from the input
                                to output dimensionality, or a more expressive MLP using multiple linear+activation+layer_norm+dropout layers
                - use_norm --> Boolean, will affect if LayerNorm is used or not in the FinetuningPredictionHead (Only matters if head_arch is set to mlp)
                - mlp_hidden_dims --> A list of the hidden dimensions for the MLP head (E.g. [512, 512, 512]). The number of the linear layers will be len(mlp_hidden_dims)+1
                - beta1 --> Value of the beta1 parameter as used by AdamW optimizer; Typically, set to 0.9
                - beta2 --> Value of the beta2 paramater as used by AdamW optimizer; Typically, set to 0.999
                - head_activation --> Activation to be used in the TaskHead; By default, it is nn.GELU
                - head_dropout --> Dropout probability for TaskHead; By default it is 0.2.
        """
        super().__init__()

        self.task_type = task_type
        self.num_tasks = num_tasks
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.base_model_class = base_model_class
        self.finetuning_args = fix_finetuning_args(finetuning_args)
        self.use_multiple_optimizers = self.finetuning_args.get(
            "optimizer_args", {}
        ).get("use_multiple_optimizers", False)

        self.save_hyperparameters()

        # Initialize the BaseModel and load the passed in checkpoint or the default pretrained:
        logging.info(f"Initializing model with params {model_params}")
        model_class = FineTuneLightningModule.import_class(base_model_class)
        self.model = model_class(**model_params)

        if checkpoint_path is None:
            logging.info(
                "Initialized the model with random weights as checkpoint_path was specified as None"
            )
        else:
            logging.info(f"Loading model from given checkpoint_path {checkpoint_path}")
            self.model.load_ckpt(checkpoint_path)

        # For indexing into the batch to get labels
        self.batch_index = modality_map[self.model.modality]
        # For storing the respective uni-modal coefficients during test steps
        if self.model.modality is Modality.MULTIVIEW:
            self.modality_coeffs_test = []

        # Creating the Prediction Head for the Finetuning Task
        logging.info(f"Task type is {task_type}")
        self.pred_head = MultiTaskPredictionHead(
            input_dim=self.model.get_embed_dim(),
            num_tasks=num_tasks,
            task_type=TaskType(task_type.lower()),
            head=finetuning_args.get("head_arch", "mlp"),
            hidden_dims=finetuning_args.get("mlp_hidden_dims", [512, 384]),
            use_norm=finetuning_args.get("use_norm", True),
            activation=finetuning_args.get("head_activation", nn.GELU),
            dropout_prob=finetuning_args.get("head_dropout", 0.2),
            bias_terms=finetuning_args.get("bias_terms", []),
        )
        # Handles the initialization of the prediction head based on the user config
        self.initialization = finetuning_args.get("initialization", "default")
        FineTuneLightningModule.init_head_weights(self.pred_head, self.initialization)

        # Handles the freezing of the base model weights based on the user config
        self.weight_freeze = finetuning_args.get("weight_freeze")
        self.base_model_frozen = False
        self.freeze_weights()
        logging.info(
            f"Base model weights frozen at start of training: {self.base_model_frozen}"
        )
        if self.model.modality == Modality.MULTIVIEW:
            self.model.change_agg_freeze()

        # Learning rate pattern
        self.learning_rate_policy = finetuning_args.get("scheduler")

        self._initialize_metrics()

        class_weights = finetuning_args.get("class_weights", None)

        try:
            if class_weights:
                self.class_weights = np.array(list(class_weights))
                logging.info(f"Using class weights: {self.class_weights}")
            else:
                self.class_weights = None
                logging.info("No class weights provided. Using None for class weights.")
        except (TypeError, ValueError) as e:
            logging.error(
                f"Invalid class weights provided: {class_weights}. Error: {e}"
            )
            self.class_weights = None
            logging.info("Using None for class weights due to error.")

        # Storage for test results
        self.test_results = {"predictions": [], "labels": []}
        logging.info(f"Global seed: {os.environ.get(ns.ENV_RANDOM_SEED)}")
        if self.use_multiple_optimizers:
            self.automatic_optimization = False
            logging.info(
                f"Using multiple optimizers with configuration {self.finetuning_args.get('optimizer_args', {})}"
            )

    def _initialize_metrics(self):
        (
            self.metrics,
            self.criterion,
        ) = FineTuneLightningModule._initialize_metrics_for_task(
            self.task_type, self.num_tasks
        )

    def _initialize_metrics_for_task(task_type, num_tasks):
        metrics = {"train": {}, "val": {}, "test": {}}

        if task_type == TaskType.CLASSIFICATION.value:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            task = "multilabel" if num_tasks > 1 else "binary"
            num_labels = max(1, num_tasks)
            average = "macro"

            metrics_for_task = [
                (
                    "rocauc",
                    torchmetrics.AUROC,
                    {
                        "task": task,
                        "num_labels": num_labels,
                        "average": average,
                        "ignore_index": -1,
                    },
                ),
                (
                    "precision",
                    torchmetrics.Precision,
                    {
                        "task": task,
                        "num_labels": num_labels,
                        "average": average,
                        "ignore_index": -1,
                    },
                ),
                (
                    "recall",
                    torchmetrics.Recall,
                    {
                        "task": task,
                        "num_labels": num_labels,
                        "average": average,
                        "ignore_index": -1,
                    },
                ),
                (
                    "specificity",
                    torchmetrics.Specificity,
                    {
                        "task": task,
                        "num_labels": num_labels,
                        "average": average,
                        "ignore_index": -1,
                    },
                ),
            ]
        elif task_type == TaskType.REGRESSION.value:
            criterion = nn.MSELoss()
            metrics_for_task = [
                ("rmse", torchmetrics.regression.MeanSquaredError, {"squared": False}),
                ("mae", torchmetrics.MeanAbsoluteError, {}),
            ]
        for stage in ["train", "val", "test"]:
            for metric_name, metric_class, params in metrics_for_task:
                metrics[stage][metric_name] = metric_class(**params)

        return metrics, criterion

    def setup(self, stage):
        if (
            "Text" in self.base_model_class
            or "SmallMoleculeMultiViewModel" in str(self.base_model_class)
        ) and self.weight_freeze == "frozen":
            logging.info(
                f"Adding deterministic_eval as True - stage {stage}, weight_freeze {self.weight_freeze}"
            )
            if "Text" in self.base_model_class:
                self.model.eval()
            elif "SmallMoleculeMultiViewModel" in str(self.base_model_class):
                self.model.model_text.eval()

        device = self.device
        for stage in ["train", "val", "test"]:
            for _, metric in self.metrics[stage].items():
                metric.to(device)

    def forward(self, batch):
        if self.model.modality == Modality.MULTIVIEW:
            logits, coeffs = self.model.forward0(batch)
            return self.pred_head(logits), coeffs
        else:
            return self.pred_head(self.model.forward0(batch))

    def state_dict(self):
        if self.weight_freeze == "lora":
            return {
                "lora_base_model": lora.lora_state_dict(self.model),
                "lora_head": self.pred_head.state_dict(),
            }
        else:
            return super().state_dict()

    def _compute_loss(self, y_hat, y, step_type):
        dtype = y_hat.dtype
        device = y_hat.device
        if self.task_type == TaskType.CLASSIFICATION.value:
            valid_mask = y != -1  # [B, T]
            if torch.sum(valid_mask) == 0:
                return torch.tensor(0.0, dtype=dtype, device=device)

            y_valid = torch.where(
                valid_mask,
                y.to(dtype),
                torch.zeros_like(y, dtype=dtype),
            )  # [B, T]

            loss_mat = self.criterion(y_hat, y_valid)  # [B, T]

            loss_mat = torch.where(
                valid_mask, loss_mat, torch.zeros_like(loss_mat)
            )  # [B, T]

            if self.class_weights is not None:
                class_weights = torch.tensor(self.class_weights, dtype=dtype).to(
                    device
                )  # [T, 2]
                class_weights = class_weights[: y.shape[1], :]  # [T, 2]

                cls_weights_0 = (
                    class_weights[:, 0].unsqueeze(0).expand(y.shape[0], -1)
                )  # [B, T]
                cls_weights_1 = (
                    class_weights[:, 1].unsqueeze(0).expand(y.shape[0], -1)
                )  # [B, T]

                cls_weights = torch.where(
                    y == 1, cls_weights_1, cls_weights_0
                )  # [B, T]
                cls_weights = torch.where(
                    valid_mask, cls_weights, torch.zeros_like(cls_weights)
                )  # [B, T]

                loss = torch.sum(loss_mat * cls_weights)
                if torch.sum(valid_mask) > 0:
                    loss = loss / torch.sum(valid_mask)
            else:
                loss = torch.sum(loss_mat)
                if torch.sum(valid_mask) > 0:
                    loss = loss / torch.sum(valid_mask)

        elif self.task_type == TaskType.REGRESSION.value:
            loss = self.criterion(y_hat, y)

        self.log(f"{step_type}_loss", loss, prog_bar=True, logger=True)
        return loss

    def _shared_step(self, batch, step_type):
        y = batch[self.batch_index]
        y_hat = self(batch)
        if self.model.modality == Modality.MULTIVIEW:
            y_hat = self.multiview_extra_logging(y_hat, step_type)

        if y.ndim == 0:
            y = y.unsqueeze(dim=0)
        y_hat = y_hat.reshape(y.shape)
        if self.task_type == "classification":
            y = y.to(torch.int32)
        elif self.task_type == "regression":
            y = y.to(torch.float32)

        self._update_metrics(y_hat, y, step_type)
        return self._compute_loss(y_hat, y, step_type), y, y_hat

    def _update_metrics(self, y_hat, y, step_type):
        for metric_name, metric in self.metrics[step_type].items():
            metric.update(y_hat, y)

    def training_step(self, batch, batch_idx):
        self.freeze_weights()
        if self.model.modality == Modality.MULTIVIEW:
            self.model.change_agg_freeze(self.current_epoch)
        loss, _, _ = self._shared_step(batch, "train")
        if self.use_multiple_optimizers:
            optimizers = self.optimizers()
            self._zero_grad(optimizers)
            self.manual_backward(loss)
            self._clip_gradients(optimizers)
            self._step_optimizers(optimizers)
        return loss

    def _zero_grad(self, optimizers):
        for opt in optimizers:
            opt.zero_grad()

    def _clip_gradients(self, optimizers):
        clipping_value = self.finetuning_args.get("optimizer_args", {}).get(
            "clipping_value", 1.0
        )
        for opt in optimizers:
            self.clip_gradients(
                opt, gradient_clip_val=clipping_value, gradient_clip_algorithm="norm"
            )

    def _step_optimizers(self, optimizers):
        for opt in optimizers:
            opt.step()

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch, "test")
        self.test_results["predictions"].extend(y_hat.detach().cpu().numpy())
        self.test_results["labels"].extend(y.detach().cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")

    def on_validation_epoch_end(self):
        self._log_and_reset_metrics("val")

    def on_test_epoch_end(self):
        self._log_and_reset_metrics("test")
        self.save_test_results()
        if self.model.modality == Modality.MULTIVIEW:
            self.multiview_extra_logging(None, "test", end_of_test=True)

    def _log_and_reset_metrics(self, stage):
        for metric_name, metric in self.metrics[stage].items():
            self.log(f"{stage}_{metric_name}", metric.compute())
            metric.reset()

    def save_test_results(self):
        df = pd.DataFrame(
            {
                "prediction": self.test_results["predictions"],
                "label": self.test_results["labels"],
            }
        )
        log_version = os.path.basename(self.trainer.log_dir)
        csv_path = os.path.join(
            self.trainer.default_root_dir, f"test_predictions_{log_version}.csv"
        )
        logging.info(f"Writing test predictions to {csv_path}")
        df.to_csv(csv_path, index=False)
        self.test_results = {"predictions": [], "labels": []}

    def multiview_extra_logging(self, multiview_output, stage, end_of_test=False):
        # Ordering is important; it matches the SMMV model
        modalities = ["Graph2dModel", "ImageModel", "TextModel"]

        if end_of_test:  # Saving the modality coefficients at the end of the Test epoch
            if len(self.modality_coeffs_test) > 0:
                df = pd.DataFrame(
                    self.modality_coeffs_test,
                    columns=[
                        modalities[i] + "_coeff"
                        for i in range(len(self.modality_coeffs_test[0]))
                    ],
                )
                log_version = os.path.basename(self.trainer.log_dir)
                modality_coeffs_csv_path = os.path.join(
                    self.trainer.default_root_dir,
                    f"modality_coeff_test_{log_version}.csv",
                )
                average_values = df.mean().to_frame().T
                df = pd.concat([df, average_values], ignore_index=True)
                df.insert(0, "mini_batch", list(range(len(df) - 1)) + ["average"])
                df.to_csv(modality_coeffs_csv_path, index=False)
                logging.info(
                    f"Writing test modality coefficients to {modality_coeffs_csv_path}"
                )
            return

        # Logging the modality coefficients during the train/val/test steps
        y_hat, model_coeffs = multiview_output
        if model_coeffs is not None:
            model_coeffs = model_coeffs.clone().detach().cpu().tolist()
            for i in range(len(model_coeffs)):
                self.log(
                    "modality_coefficients_" + stage + "/" + modalities[i],
                    model_coeffs[i],
                )
            self.modality_coeffs_test.append(model_coeffs)
        return y_hat

    def on_before_optimizer_step(self, optimizer):
        norms_ph = grad_norm(self.pred_head, norm_type=2)
        self.log_dict(norms_ph)

    def configure_optimizers(self):
        def configure_single_optimizer():
            """
            Possible params needed:
            - T_max --> Cosine anneal
            - Max_LR --> One_cycle
            - power --> polynomial
            - step_size/gamma --> Step.
            """
            beta1 = self.finetuning_args.get("beta1", 0.9)
            beta2 = self.finetuning_args.get("beta2", 0.999)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(beta1, beta2),
            )

            scheduler = None

            # Follows a cosine pattern for LR, starting at the max lr, t_max is number of epochs for 1 cosine cycle (before going back to max_lr again)
            if self.learning_rate_policy == "cosine_anneal":
                print("Created the cosine anneal scheduler")
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.finetuning_args.get("t_max", 100)
                )
            # Anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate
            elif self.learning_rate_policy == "one_cycle":
                print("Created the one cycle LR scheduler")
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.finetuning_args.get("max_lr", 0.1),
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                )
            # Decays the learning rate using a polynomial function over the given total_iters
            elif self.learning_rate_policy == "poly":
                print("Created the polynomial LR scheduler")
                scheduler = optim.lr_scheduler.PolynomialLR(
                    optimizer,
                    total_iters=self.trainer.max_epochs,
                    power=0.9,
                    last_epoch=-1,
                )
            elif (
                self.learning_rate_policy == "step"
            ):  # Reduces learning_rate by gamma every step_size no. of epochs
                print("Created the step lr scheduler")
                scheduler = scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=40, gamma=0.1
                )
            else:  # Not using a scheduler
                return optimizer

            return [optimizer], [scheduler]

        def get_optimizer_args(modality, arg):
            if self.finetuning_args["optimizer_args"].get(modality, None) is not None:
                if (
                    self.finetuning_args["optimizer_args"][modality].get(arg)
                    is not None
                ):
                    arg_val = self.finetuning_args["optimizer_args"][modality][arg]
            return arg_val

        if self.use_multiple_optimizers:

            def get_lr(model_name, default_lr):
                lr = get_optimizer_args(model_name, "lr")
                return float(lr) if lr is not None else default_lr

            text_lr = get_lr(ns.Models.TEXT_MODEL.name, 3e-5)
            image_lr = get_lr(ns.Models.IMAGE_MODEL.name, 1e-4)
            graph_lr = get_lr(ns.Models.GRAPH_2D_MODEL.name, 1e-5)
            rest_lr = get_lr("REST", 1e-4)

            text_opt = torch.optim.Adam(self.model.model_text.parameters(), lr=text_lr)
            graph_opt = torch.optim.Adam(
                self.model.model_graph.parameters(), lr=graph_lr
            )
            image_opt = torch.optim.SGD(
                self.model.model_image.parameters(),
                lr=image_lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            rest_opt = torch.optim.Adam(
                [
                    {"params": self.model.aggregator.parameters()},
                    {"params": self.pred_head.parameters()},
                ],
                lr=rest_lr,
            )
            return [text_opt, graph_opt, image_opt, rest_opt]
        else:
            return configure_single_optimizer()

    # HELPER FUNCTIONS FOR FINETUNING STRATEGIES
    def _change_grad(self, model, requires_grad):
        for name, param in model.named_parameters():
            param.requires_grad = requires_grad

    def _freeze_weights(self, weight_freeze, model, isFrozen):
        if weight_freeze == "frozen" and (isFrozen is None or isFrozen is False):
            self._change_grad(model, False)
            return True
        elif weight_freeze == "unfrozen" and (isFrozen is None or isFrozen):
            self._change_grad(model, True)
            return False
        elif weight_freeze == "gradual":
            if self.current_epoch == 0:
                if isFrozen is None or isFrozen is False:
                    self._change_grad(model, False)
                    return True
            else:
                if isFrozen is None or isFrozen:
                    self._change_grad(model, True)
                    return False
        return isFrozen

    def freeze_weights(self):
        if self.weight_freeze in ["frozen", "unfrozen", "gradual"]:
            self.base_model_frozen = self._freeze_weights(
                self.weight_freeze, self.model, self.base_model_frozen
            )
        elif self.weight_freeze == "lora" and self.base_model_frozen is not True:
            # FineTuneLightningModule.replace_with_lora(self.pred_head) #Testing if prediction head should be trained fully (not Lora approximation)
            FineTuneLightningModule.replace_with_lora(self.model)
            self.base_model_frozen = True

    """
    How to use Lora:
    - The lora layers add extra parameters (LoraA and loraB) that the checkpoint will then save
    - However, when restoring from a checkpoint, you can't only restore the lora parameters, you have to restore both the regular (pretrained) checkpoint and THEN the lora checkpoint
        as opposed to just restoring one checkpoint
    """

    @staticmethod
    def replace_with_lora(module, rank=2):
        for name, child in module.named_children():
            if isinstance(child, nn.modules.conv.Conv2d):
                setattr(
                    module,
                    name,
                    lora.Conv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size[0],
                        r=rank,
                        stride=child.stride,
                        padding=child.padding,
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    lora.Linear(child.in_features, child.out_features, r=rank),
                )
            elif len(list(child.children())) == 0:
                child.requires_grad = False
            else:
                FineTuneLightningModule.replace_with_lora(child)

        if torch.cuda.is_available():
            module.to(torch.device("cuda"))

    @staticmethod
    def init_head_weights(module, initialization="default"):
        last_bias = None
        for name, layer in module.named_modules():
            if isinstance(layer, nn.Linear):
                if initialization == "xavier":
                    nn.init.xavier_normal_(layer.weight)
                elif initialization == "he":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif initialization == "default":
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                else:
                    raise ValueError(
                        f"Initialization strategy {initialization} is not supported."
                    )
                last_bias = layer.bias.detach().tolist()

        if last_bias is not None:
            logging.info(
                f"Biases in the last layer after initializing head weights: {last_bias}"
            )

    # UTILITY FUNCTIONS

    @staticmethod
    def import_class(class_path):
        if type(class_path) is str:
            split_path = class_path.rsplit(".", 1)
            module_name, class_name = split_path[0], split_path[1]
            return getattr(importlib.import_module(module_name), class_name)
        else:
            return class_path
