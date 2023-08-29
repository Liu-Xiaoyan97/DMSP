#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 21:13
# @Author  : lxy15058247683@aliyun.com
# @FileName: utils.py
# @Copyright: MIT
import torch


def token_transform(tokens: torch.Tensor, times: int, embedding: torch.nn.Embedding):
    batch, length = tokens.size()
    copy_tensor = tokens.clone()
    shuffle_tensors = []
    for _ in range(times):
        shuffle_tensors.append(embedding(copy_tensor[:, torch.randperm(length)]))
    return shuffle_tensors
