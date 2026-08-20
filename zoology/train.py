import argparse
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
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
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
        self.slice_keys = slice_keys
        self.loss_type = loss_type

    def compute_loss(self, inputs, targets):
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
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs, return_embeddings=False)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
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
                return loss, preds

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

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
            self.optimizer.step()
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log({"train/loss": loss.item(), "epoch": epoch_idx})

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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

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
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

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
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    task.fit()
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
