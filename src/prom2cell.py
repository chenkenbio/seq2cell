#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2023-02-25
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

ONEHOT = torch.cat((
    torch.ones(1, 4) / 4, # 
    torch.eye(4), # A, C, G, T
    torch.zeros(1, 4), # padding
), dim=0).float()

class LayerNorm(nn.Module):
    """ 
    taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py#L79

    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

# class ResBlock(nn.Module):
#     def __init__(self, kernel_num, kernel_size) -> None:
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(kernel_num, kernel_num, kernel_size, padding=kernel_size//2),
#             LayerNorm(kernel_num),
#             nn.ReLU(),
#             nn.Conv1d(kernel_num, kernel_num, kernel_size, padding=kernel_size//2),
#             LayerNorm(kernel_num),
#             nn.ReLU(),
#         )
    
#     def forward(self, x: Tensor) -> Tensor:
#         residual = x
#         x = self.conv(x)
#         return x + residual

class ConvBlock(nn.Module):
    def __init__(self, kernel_num, kernel_size, pooling: bool=True) -> None:
        super().__init__()
        convs = [
            nn.Conv1d(kernel_num, kernel_num, kernel_size, padding=kernel_size//2),
            LayerNorm(kernel_num),
            nn.ReLU(),
        ]
        if pooling:
            convs.append(nn.MaxPool1d(kernel_size//2))
        self.convs = nn.Sequential(*convs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x)

class ResBlock(nn.Module):
    def __init__(self, kernel_num, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(kernel_num, kernel_size, pooling=False),
            ConvBlock(kernel_num, kernel_size, pooling=False),
        )
        self.pool = nn.MaxPool1d(kernel_size//2)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv(x)
        return self.pool(x + residual)

# seq_len = 65536
class SequenceEncoder(nn.Module):
    def __init__(self, kernel_num=256, seq_len: int=512 * 128) -> None:
        super().__init__()
        self.onehot = nn.Parameter(ONEHOT, requires_grad=False)
        self.pre_conv = nn.Sequential(
            nn.Conv1d(4, kernel_num, 17, padding=8),
            LayerNorm(kernel_num),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
        ) # (batch_size, kernel_num, seq_len / 8)
        convs = []
        for i in range(3):
            convs.append(ResBlock(kernel_num, 9))
        self.convs = nn.Sequential(*convs)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.onehot[x].transpose(1, 2)
        x = self.pre_conv(x)
        x = self.convs(x)
        return x
            


from torch.distributions import NegativeBinomial

class GenePredictor(nn.Module):
    def __init__(self, bottleneck: int, n_cells: int, gene_ids: Union[Tensor, np.ndarray]) -> None:
        super(GenePredictor, self).__init__()
        assert gene_ids.ndim == 1, "gene_ids should be 1D, but got shape: {}".format(gene_ids.shape)
        assert len(np.unique(gene_ids)) == len(gene_ids), "gene_ids should be unique, but got {} non-unique elements".format(len(gene_ids) - len(np.unique(gene_ids)))
        assert max(gene_ids) == len(gene_ids) - 1, "gene_ids should be 0-indexed and continuous in range [0, len(gene_ids) - 1], but got max(gene_ids) = {}".format(np.max(gene_ids))
    
    def forward(self, embedding: Tensor, gene_ids: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_loss(self, embedding, gene_ids, obs):
        raise NotImplementedError


class NBPredictor(GenePredictor):
    def __init__(self, bottleneck: int, n_cells: int, gene_ids: Union[Tensor, np.ndarray]) -> None:
        super().__init__(bottleneck, n_cells, gene_ids)
        self.counts_k = nn.Embedding(len(gene_ids), 1)
        self.counts_b = nn.Embedding(len(gene_ids), 1)
        self.logits_k = nn.Embedding(len(gene_ids), 1)
        self.logits_b = nn.Embedding(len(gene_ids), 1)

        self.peak2logits = nn.Linear(bottleneck, n_cells)
        self.peak2counts = nn.Linear(bottleneck, n_cells)
    
    def forward(self, embedding: Tensor, gene_ids: Tensor) -> Tensor:

        seq2counts = self.peak2counts(embedding) # (batch_size, n_cells)
        seq2logits = self.peak2logits(embedding) # (batch_size, n_cells)

        counts = self.counts_k(gene_ids) * seq2counts + self.counts_b(gene_ids) # (batch_size, n_cells)
        logits = self.logits_k(gene_ids) * seq2logits + self.logits_b(gene_ids)
        counts = F.softplus(counts)

        return NegativeBinomial(total_count=counts, logits=logits)
    
    def get_loss(self, embedding, gene_ids, obs: Tensor):
        loss = - self.forward(embedding, gene_ids).log_prob(obs).mean()
        return loss


class Peak2Cell(nn.Module):
    def __init__(self, n_cells, encoder: nn.Module, gene_ids: Union[Tensor, np.ndarray], batch_ids: Optional[Tensor]=None, bottleneck_size: int=32) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = NBPredictor(
            bottleneck=bottleneck_size,
            n_cells=n_cells,
            gene_ids=gene_ids,
        )
    
    def forward(self, sequence: Tensor, gene_ids: Tensor) -> Tensor:

        assert sequence.shape[0] == gene_ids.shape[0], "sequence.shape[0] = {}, gene_ids.shape[0] = {}".format(sequence.shape[0], gene_ids.shape[0]) # each sample is a gene

        sequence = self.encoder(sequence) # (batch_size, hidden_size), batch_size: genes
        return self.decoder(sequence, gene_ids)


