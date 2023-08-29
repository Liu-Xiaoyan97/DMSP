#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 21:58
# @Author  : lxy15058247683@aliyun.com
# @FileName: MSP_lightning.py
# @Copyright: MIT
from typing import Any, Tuple, Optional, Union, List

import numpy as np
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn
from transformers import AutoTokenizer

shuffle_times = 3


from utils import token_transform


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, hidden_dim),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SemanticPerceptron(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.mlp_layer = MultiLayerPerceptron(hidden_dim, ffn_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor, shuffle_tensors: List[torch.Tensor]) -> torch.Tensor:
        residual = inputs
        for shuffle_tensor in shuffle_tensors:
            res = self.dropout(self.layer_norm(self.mlp_layer(inputs)+residual-shuffle_tensor))
        res = self.layer_norm(res+residual)
        return res


class MultiSemanticPerceptron(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, ffn_dim: int,
                 num_layers: int, num_classes: int, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.msp = nn.ModuleList(
            [SemanticPerceptron(hidden_dim, ffn_dim) for i in range(num_layers)]
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor, shuffle_times: int) -> torch.Tensor:
        res = self.embedding(inputs)
        for sp in self.msp:
            shuffle_tokens = token_transform(inputs, shuffle_times, self.embedding)
            res = sp(res, shuffle_tokens)
        res = res.mean(dim=1)
        logit = self.cls(res)
        return logit


class ModelModule(pl.LightningModule):
    # vocab_size: int, embedding_dim: int, hidden_dim: int, ffn_dim: int,
    #                  num_layers: int, num_classes: int
    def __init__(self, vocab_size: int, hidden_dim: int, ffn_dim: int,
                 num_layers: int, num_classes: int, lr: float,
                 **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.msp = MultiSemanticPerceptron(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **kwargs)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return [opt], []

    def share_step(self, inputs: np.array, targets: np.array, shuffle_times: int) -> Any:
        logit = self.msp(inputs, shuffle_times)
        loss = self.criterion(logit, targets.long())
        return logit, loss

    @staticmethod
    def covert_logit_to_idx(logit) -> torch.Tensor:
        return torch.argmax(logit, dim=-1)

    @staticmethod
    def compute_accuracy(prediction, ground_truth):
        return torch.sum((prediction == ground_truth))/(ground_truth.size(0) + 1e-9)

    def log_step(self, stage, chunk, loss, accuracy, prog_bar, on_step, on_epoch):
        self.log(f"{stage}_{chunk}_loss", loss, prog_bar, on_step, on_epoch)
        self.log(f"{stage}_{chunk}_accuracy", accuracy, prog_bar, on_step, on_epoch)

    def average_metrics(self, outputs: EPOCH_OUTPUT, stage: str):
        epoch_accuracy = torch.stack([output['accuracy'] for output in outputs]).mean()
        epoch_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log_step(stage, "epoch", epoch_loss, epoch_accuracy, True, False, False)

    def training_step(self, batch: Tuple, batch_idx) -> STEP_OUTPUT:
        src, target = batch
        logit, loss = self.share_step(src, target, shuffle_times)
        prediction = self.covert_logit_to_idx(logit)
        accuracy = self.compute_accuracy(prediction, target)
        self.log_step("train", "step", loss, accuracy, True, True, True)
        return {
            "accuracy": accuracy,
            "loss": loss
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.average_metrics(outputs, "train")

    def validation_step(self, batch: Tuple, batch_idx) -> Optional[STEP_OUTPUT]:
        src, target = batch
        logit, loss = self.share_step(src, target, shuffle_times)
        prediction = self.covert_logit_to_idx(logit)
        accuracy = self.compute_accuracy(prediction, target)
        self.log_step("validation", "step", loss, accuracy, True, True, False)
        return {
            "accuracy": accuracy,
            "loss": loss
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.average_metrics(outputs, "validation")

    def test_step(self, batch: Tuple, batch_idx) -> Optional[STEP_OUTPUT]:
        src, target = batch
        logit, loss = self.share_step(src, target, shuffle_times)
        prediction = self.covert_logit_to_idx(logit)
        accuracy = self.compute_accuracy(prediction, target)
        self.log_step("test", "step", loss, accuracy, True, True, False)
        return {
            "accuracy": accuracy,
            "loss": loss
        }

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.average_metrics(outputs, "test")