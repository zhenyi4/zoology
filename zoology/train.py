import argparse
import json
import random
from datetime import datetime
from typing import List, Union
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data, prepare_continuous_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings
from zoology.diagnostics import (
    model_gradients_sha256,
    model_parameters_sha256,
    named_tensors_sha256,
    rng_state_sha256,
    runtime_diagnostics,
    tensor_sha256,
)


def log_diagnostics(logger: WandbLogger, diagnostics: dict):
    """Print diagnostics and persist them as W&B summary fields."""
    payload = {
        f"diagnostics/{key}": value
        for key, value in diagnostics.items()
    }
    print("Repeatability diagnostics:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    logger.log_summary(payload)


def build_adamw_parameter_groups(
    model: nn.Module,
    weight_decay: float,
):
    """Split trainable parameters into AdamW decay and no-decay groups.

    Biases, normalization/scalar/vector parameters, and parameters explicitly
    marked with ``_no_weight_decay`` should not be regularized. The last case
    is required by recurrent mixers such as Mamba2 and Gated DeltaNet for
    ``A_log``, ``dt_bias``, and ``D``. Using ``parameter.ndim < 2`` also covers
    LayerNorm/RMSNorm scale parameters without depending on a particular norm
    implementation.
    """
    decay_parameters = []
    no_decay_parameters = []
    decay_names = []
    no_decay_names = []
    explicitly_excluded_names = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        explicitly_excluded = getattr(parameter, "_no_weight_decay", False)
        exclude_from_weight_decay = (
            explicitly_excluded
            or name.endswith(".bias")
            or parameter.ndim < 2
        )
        if exclude_from_weight_decay:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
            if explicitly_excluded:
                explicitly_excluded_names.append(name)
        else:
            decay_parameters.append(parameter)
            decay_names.append(name)

    grouped_parameter_ids = {
        id(parameter)
        for parameter in decay_parameters + no_decay_parameters
    }
    trainable_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if grouped_parameter_ids != trainable_parameter_ids:
        missing = len(trainable_parameter_ids - grouped_parameter_ids)
        duplicated = (
            len(decay_parameters)
            + len(no_decay_parameters)
            - len(grouped_parameter_ids)
        )
        raise RuntimeError(
            "AdamW parameter grouping must cover every trainable parameter "
            f"exactly once; missing={missing}, duplicated={duplicated}."
        )

    groups = []
    if decay_parameters:
        groups.append(
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            }
        )
    if no_decay_parameters:
        groups.append(
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            }
        )

    summary = {
        "decay_parameter_tensors": len(decay_parameters),
        "decay_parameter_scalars": sum(
            parameter.numel() for parameter in decay_parameters
        ),
        "no_decay_parameter_tensors": len(no_decay_parameters),
        "no_decay_parameter_scalars": sum(
            parameter.numel() for parameter in no_decay_parameters
        ),
        "decay_parameter_names": decay_names,
        "no_decay_parameter_names": no_decay_names,
        "explicitly_excluded_parameter_names": explicitly_excluded_names,
    }
    return groups, summary


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        optimizer_parameter_grouping: str = "matrix_only",
        learning_rate_schedule: str = "cosine",
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
        diagnostic_steps: int = 0,
        max_steps: int = 0,
        log_train_accuracy: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_parameter_grouping = optimizer_parameter_grouping
        self.learning_rate_schedule = learning_rate_schedule
        self.slice_keys = slice_keys
        self.loss_type = loss_type
        if diagnostic_steps < 0:
            raise ValueError(
                f"diagnostic_steps must be non-negative; got {diagnostic_steps}."
            )
        self.diagnostic_steps = diagnostic_steps
        if max_steps < 0:
            raise ValueError(f"max_steps must be non-negative; got {max_steps}.")
        if diagnostic_steps > 0 and max_steps > 0:
            raise ValueError(
                "diagnostic_steps and max_steps cannot both be positive."
            )
        self.max_steps = max_steps
        self.log_train_accuracy = log_train_accuracy
        self.train_steps_completed = 0
        self.peak_train_accuracy = None
        self.peak_train_accuracy_step = None

    def compute_loss(self, inputs, targets, return_outputs: bool = False):
        if self.input_type == "continuous":
            
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates
            
            outputs = self.model(inputs, return_embeddings=True)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]
            
            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)
            
            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )
            
            logits = outputs_flat @ value_embeddings.T
            preds = (logits).argmax(dim=-1).view(targets.shape)
            if return_outputs:
                return loss, preds, logits
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs, return_embeddings=False)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
                if return_outputs:
                    return loss, preds, logits
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                preds = logits.argmax(dim=-1)
                if return_outputs:
                    return loss, preds, logits
                return loss, preds
            
            elif self.loss_type == "ce_embed":
                embeddings = self.model(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                logits = embeddings @ value_embeddings.T
                preds = logits.argmax(dim=-1)
                if return_outputs:
                    return loss, preds, logits
                return loss, preds

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            diagnostic_step = self.train_steps_completed < self.diagnostic_steps
            diagnostics = {}
            if diagnostic_step:
                step_prefix = f"step_{self.train_steps_completed}"
                diagnostics.update(
                    {
                        f"{step_prefix}/batch_inputs_sha256": tensor_sha256(
                            inputs, name="inputs"
                        ),
                        f"{step_prefix}/batch_targets_sha256": tensor_sha256(
                            targets, name="targets"
                        ),
                        f"{step_prefix}/batch_sha256": named_tensors_sha256(
                            [("inputs", inputs), ("targets", targets)]
                        ),
                    }
                )
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            if diagnostic_step:
                diagnostics[f"{step_prefix}/rng_before_forward_sha256"] = (
                    rng_state_sha256()
                )
                loss, preds, model_outputs = self.compute_loss(
                    inputs,
                    targets,
                    return_outputs=True,
                )
                outputs_to_hash = model_outputs
                if (
                    model_outputs.ndim == targets.ndim + 1
                    and model_outputs.shape[:-1] == targets.shape
                ):
                    # Full MQAR logits can exceed 2 GB at batch size 256.
                    # Answer-position logits are sufficient to detect the
                    # first numerical divergence and are inexpensive to hash.
                    outputs_to_hash = model_outputs[targets != -100]
                diagnostics.update(
                    {
                        f"{step_prefix}/model_outputs_sha256": tensor_sha256(
                            outputs_to_hash, name="supervised_model_outputs"
                        ),
                        f"{step_prefix}/predictions_sha256": tensor_sha256(
                            preds, name="predictions"
                        ),
                        f"{step_prefix}/loss_sha256": tensor_sha256(
                            loss, name="loss"
                        ),
                        f"{step_prefix}/loss_value": loss.item(),
                        f"{step_prefix}/rng_after_forward_sha256": (
                            rng_state_sha256()
                        ),
                    }
                )
            else:
                loss, preds = self.compute_loss(inputs, targets)

            # Auxiliary losses (discrete mode only)
            if self.input_type == "discrete":
                auxiliary_loss = []
                def get_auxiliary_loss(module):
                    if hasattr(module, "get_auxiliary_loss"):
                        auxiliary_loss.append(module.get_auxiliary_loss())
                self.model.apply(get_auxiliary_loss)
                if auxiliary_loss:
                    loss = loss + sum(auxiliary_loss)

            loss.backward()
            if diagnostic_step:
                diagnostics.update(
                    {
                        f"{step_prefix}/gradients_sha256": (
                            model_gradients_sha256(self.model)
                        ),
                        f"{step_prefix}/rng_after_backward_sha256": (
                            rng_state_sha256()
                        ),
                    }
                )
            self.optimizer.step()
            if diagnostic_step:
                diagnostics[f"{step_prefix}/model_post_step_sha256"] = (
                    model_parameters_sha256(self.model)
                )
                log_diagnostics(self.logger, diagnostics)

            self.train_steps_completed += 1
            train_metrics = {
                "train/loss": loss.item(),
                "train/step": self.train_steps_completed,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch_idx,
            }
            if self.log_train_accuracy:
                supervised = targets != -100
                if supervised.any():
                    train_accuracy = (
                        (preds[supervised] == targets[supervised])
                        .to(torch.float32)
                        .mean()
                        .item()
                    )
                    train_metrics["train/accuracy"] = train_accuracy
                    if (
                        self.peak_train_accuracy is None
                        or train_accuracy > self.peak_train_accuracy
                    ):
                        self.peak_train_accuracy = train_accuracy
                        self.peak_train_accuracy_step = self.train_steps_completed
                    train_metrics["train/peak_accuracy"] = self.peak_train_accuracy
            iterator.set_postfix(
                {
                    "loss": train_metrics["train/loss"],
                    **(
                        {"acc": train_metrics["train/accuracy"]}
                        if "train/accuracy" in train_metrics
                        else {}
                    ),
                }
            )
            self.logger.log(train_metrics)

            if (
                self.diagnostic_steps > 0
                and self.train_steps_completed >= self.diagnostic_steps
            ):
                print(
                    "Diagnostic step limit reached; stopping before validation "
                    f"after {self.train_steps_completed} optimizer step(s)."
                )
                return "diagnostic"
            if self.max_steps > 0 and self.train_steps_completed >= self.max_steps:
                print(
                    "Training step limit reached after "
                    f"{self.train_steps_completed} optimizer step(s)."
                )
                return "max_steps"
        return None

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0
        results = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(self.test_dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})
        return metrics

    def fit(self):
        self.model.to(self.device)
        if self.diagnostic_steps > 0:
            log_diagnostics(
                self.logger,
                {
                    "model_at_fit_start_sha256": model_parameters_sha256(
                        self.model
                    ),
                    "rng_at_fit_start_sha256": rng_state_sha256(),
                },
            )
        self.loss_fn = nn.CrossEntropyLoss()
        if self.optimizer_parameter_grouping == "matrix_only":
            optimizer_parameters, parameter_group_summary = (
                build_adamw_parameter_groups(
                    self.model,
                    weight_decay=self.weight_decay,
                )
            )
            print(
                "AdamW parameter groups: "
                f"decay={parameter_group_summary['decay_parameter_tensors']} tensors/"
                f"{parameter_group_summary['decay_parameter_scalars']} scalars at "
                f"weight_decay={self.weight_decay}; "
                f"no_decay={parameter_group_summary['no_decay_parameter_tensors']} tensors/"
                f"{parameter_group_summary['no_decay_parameter_scalars']} scalars."
            )
            if parameter_group_summary["explicitly_excluded_parameter_names"]:
                print(
                    "Parameters explicitly marked no-weight-decay: "
                    + ", ".join(
                        parameter_group_summary[
                            "explicitly_excluded_parameter_names"
                        ]
                    )
                )
            optimizer_weight_decay = 0.0
        elif self.optimizer_parameter_grouping == "uniform":
            optimizer_parameters = self.model.parameters()
            optimizer_weight_decay = self.weight_decay
            print(
                "AdamW parameter grouping: uniform legacy mode; all trainable "
                f"parameters use weight_decay={self.weight_decay}."
            )
        else:
            raise ValueError(
                "optimizer_parameter_grouping must be 'matrix_only' or "
                f"'uniform'; got {self.optimizer_parameter_grouping!r}."
            )
        self.optimizer = optim.AdamW(
            optimizer_parameters,
            lr=self.learning_rate,
            weight_decay=optimizer_weight_decay,
        )
        if self.learning_rate_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs, eta_min=0.0
            )
        elif self.learning_rate_schedule == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda _: 1.0,
            )
        else:
            raise ValueError(
                "learning_rate_schedule must be 'cosine' or 'constant'; "
                f"got {self.learning_rate_schedule!r}."
            )
        for epoch_idx in range(self.max_epochs):
            stop_reason = self.train_epoch(epoch_idx)
            if stop_reason == "diagnostic":
                break
            metrics = self.test(epoch_idx)

            if stop_reason == "max_steps":
                break

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()

        if self.peak_train_accuracy is not None:
            self.logger.log_summary(
                {
                    "train/peak_accuracy": self.peak_train_accuracy,
                    "train/peak_accuracy_step": self.peak_train_accuracy_step,
                    "train/optimizer_steps_completed": self.train_steps_completed,
                }
            )


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    if config.diagnostic_steps < 0:
        raise ValueError(
            f"diagnostic_steps must be non-negative; got {config.diagnostic_steps}."
        )
    if config.max_steps < 0:
        raise ValueError(f"max_steps must be non-negative; got {config.max_steps}.")
    if config.diagnostic_steps > 0 and config.max_steps > 0:
        raise ValueError("diagnostic_steps and max_steps cannot both be positive.")

    set_determinism(config.seed, strict=config.strict_determinism)
    setup_diagnostics = {}
    if config.diagnostic_steps > 0:
        setup_diagnostics["rng_after_seed_sha256"] = rng_state_sha256()
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()
    if config.diagnostic_steps > 0:
        setup_diagnostics["rng_after_wandb_init_sha256"] = rng_state_sha256()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        if config.diagnostic_steps > 0:
            setup_diagnostics["model_initial_sha256"] = (
                model_parameters_sha256(model)
            )
            setup_diagnostics["rng_after_model_init_sha256"] = rng_state_sha256()
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        if config.diagnostic_steps > 0:
            setup_diagnostics["model_initial_sha256"] = (
                model_parameters_sha256(model)
            )
            setup_diagnostics["rng_after_model_init_sha256"] = rng_state_sha256()
        train_dataloader, test_dataloader = prepare_data(config.data)

    if config.diagnostic_steps > 0:
        setup_diagnostics["rng_after_data_prepare_sha256"] = rng_state_sha256()
        setup_diagnostics.update(
            {
                f"runtime/{key}": value
                for key, value in runtime_diagnostics().items()
            }
        )
        log_diagnostics(logger, setup_diagnostics)

    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optimizer_parameter_grouping=config.optimizer_parameter_grouping,
        learning_rate_schedule=config.learning_rate_schedule,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        diagnostic_steps=config.diagnostic_steps,
        max_steps=config.max_steps,
        log_train_accuracy=config.log_train_accuracy,
    )
    task.fit()
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
