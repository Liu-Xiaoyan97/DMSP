#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 21:13
# @Author  : lxy15058247683@aliyun.com
# @FileName: DMSP_lightning.py
# @Copyright: MIT
import warnings
from typing import List, Any, Tuple, Optional, Union

import numpy as np
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
from transformers import AutoTokenizer
import pytorch_lightning as pl
from Modules.MSP_lightning import MultiLayerPerceptron
from utils import token_transform


shuffle_times=3


class DynamicSemanticPerceptron(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_classes: int):
        super().__init__()
        self.mlp_layer = MultiLayerPerceptron(hidden_dim, ffn_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor, shuffle_tensors: List[torch.Tensor]) -> torch.Tensor:
        residual = inputs
        for shuffle_tensor in shuffle_tensors:
            res = self.layer_norm(self.mlp_layer(inputs)+residual-shuffle_tensor)
        res = self.layer_norm(res+residual)
        logit = self.cls(res.mean(dim=1))
        return res, logit


class DynamicMultiSemanticPerceptron(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, ffn_dim: int,
                 num_layers: int, num_classes: int, epsilon: float, patience: int, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.epsilon = epsilon
        self.msp = nn.ModuleList(
            [DynamicSemanticPerceptron(hidden_dim, ffn_dim, num_classes) for i in range(num_layers)]
        )
        self.patience = patience
        self.patience_counter = 0

    @staticmethod
    def recorder(accuracy: List, epsilon: float):
        '''
        This method is called Dynamic Depth Controller
        '''
        gl, sl = [], []
        for i, acc in enumerate(accuracy):
            if acc > epsilon:
                gl.append(i)
            else:
                sl.append(i)
        n_gl = len(gl)
        n_sl = len(sl)
        mu = torch.ones([len(accuracy)])
        accuracy = torch.tensor(accuracy)
        flag = False
        if torch.any(accuracy <= epsilon):
            if n_gl >= n_sl + 1:
                flag = True
                mu = torch.where(accuracy<=epsilon, -torch.inf, mu)
            else:
                mu = torch.where(accuracy<=epsilon, -1, mu)
        beta = torch.exp(mu*accuracy)/torch.exp(mu).sum()
        weight_accuracy = beta*accuracy
        return weight_accuracy, beta, flag

    def __update__(self):
        self.msp.pop(-1)
        self.patience_counter = 0

    @staticmethod
    def compute_accuracy(prediction, ground_truth):
        return torch.sum((prediction == ground_truth)) / (ground_truth.size(0) + 1e-9)

    @staticmethod
    def covert_logit_to_idx(logit) -> torch.Tensor:
        return torch.argmax(logit, dim=-1)

    def forward(self, inputs: torch.Tensor, target: np.array, shuffle_times: int) -> torch.Tensor:
        res = self.embedding(inputs)
        results = []
        logits = []
        for i, sp in enumerate(self.msp):
            shuffle_tokens = token_transform(inputs, shuffle_times, self.embedding)
            res, logit = sp(res, shuffle_tokens)
            predict = self.covert_logit_to_idx(logit)
            accuracy = self.compute_accuracy(predict, target.long())
            results.append(accuracy)
            logits.append(logit)
        weight_accuracy, beta, flag = self.recorder(results, self.epsilon)
        if flag:
            self.patience_counter = self.patience_counter+1
        if self.patience_counter == self.patience:
            self.__update__()
            weight_accuracy = weight_accuracy[:-2]
            logits.pop(-1)
            del beta[-1]
            warnings.WarningMessage("Removed last layer, current depth =", len(self.msp))
        return weight_accuracy, logits[-1]*beta[-1]


class ModelModule(pl.LightningModule):
    # vocab_size: int, embedding_dim: int, hidden_dim: int, ffn_dim: int,
    #                  num_layers: int, num_classes: int
    def __init__(self, vocab_size: int, hidden_dim: int, ffn_dim: int,
                 num_layers: int, num_classes: int, lr: float, epsilon: float, patience: int,
                 **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.msp = DynamicMultiSemanticPerceptron(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            epsilon=epsilon,
            patience=patience,
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
        weight_accuracy, logit = self.msp(inputs, targets, shuffle_times)
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