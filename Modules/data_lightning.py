#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 22:37
# @Author  : lxy15058247683@aliyun.com
# @FileName: data_lightning.py
# @Copyright: MIT
from typing import List, Dict

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

import platform
if platform.system() == "Linux":
    num_workers = 40
else:
    num_workers = 1


configs = OmegaConf.load("config.yml")
dataset_conf, model_conf = configs.dataset, configs.model


class HuggingFaceBenchMark(Dataset):
    def __init__(self,file_path_dir: List[str], mode: str, select_class: List[str] = None,
                 padding: str="max_length", max_len: int = 64, truncation: bool = True, label_map: Dict=None):
        self.file_path_dir = file_path_dir
        self.mode = mode
        self.padding = padding
        self.max_len = max_len
        self.truncation = truncation
        if len(file_path_dir) > 1:
            subset_name = file_path_dir[1]
        else:
            subset_name = None
        mainset_name = file_path_dir[0]
        self.data = load_dataset(mainset_name, subset_name, split=mode)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        # self.tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        values = [i for i in range(len(label_map))]
        self.label_map = dict(zip(label_map, values))

    def compute_label(self, field):
        label = self.label_map[field["label"]]
        return np.array(label)

    def convert_str_to_id(self, field):
        text = field[dataset_conf.feature1]
        output = self.tokenizer(text, padding='max_length', truncation=self.truncation,
                                max_length=self.max_len, add_special_tokens=True)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        field = self.data[item]
        feature = self.convert_str_to_id(field)
        label = self.compute_label(field)
        return (
            np.array(feature["input_ids"]),
            np.array(label)
        )


class DataModuleForSentenceClassification(pl.LightningDataModule):
    def __init__(self, file_path_dir: str, batch_size: int=32, max_len: int=32, label_map = None):
        super().__init__()
        self.file_path_dir = file_path_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.label_map = label_map
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set = HuggingFaceBenchMark(self.file_path_dir, dataset_conf.train, max_len=self.max_len, label_map=self.label_map)
            self.eval_set = HuggingFaceBenchMark(self.file_path_dir, dataset_conf.validation , max_len=self.max_len, label_map=self.label_map)
        if stage == "test" or stage is None:
            self.test_set = HuggingFaceBenchMark(self.file_path_dir, dataset_conf.test, max_len=self.max_len, label_map=self.label_map)
        if stage == "predict" or stage is None:
            self.predict_set = HuggingFaceBenchMark(self.file_path_dir, "predict", max_len=self.max_len, label_map=self.label_map)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.eval_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)