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
import h5py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from biock import load_fasta, get_reverse_strand, encode_sequence
from biock import HG38_FASTA_H5

import logging
logger = logging.getLogger(__name__)


def load_adata(data) -> AnnData:
    adata = sc.read_h5ad(data)
    if adata.X.max() > 1:
        logger.info("binarized")
        adata.X.data = (adata.X.data > 0).astype(np.float32)
    return adata

class SingleCellDataset(Dataset):
    def __init__(self, data: AnnData, genome, seq_len=1344):
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))
        self.data = data
        self.seq_len = seq_len
        self.genome = h5py.File(genome, 'r')
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T)
        del self.data.X
        if "chrom" in self.var.keys():
            self.chroms = self.var["chrom"]
        elif "chr" in self.var.keys():
            self.chroms = self.var["chr"]
        if "batch" in self.obs:
            batch2index = {batch:index for index, batch in enumerate(self.obs['batch'].unique())}
            self.batche_ids = np.asarray(self.obs['batch'].map(batch2index).values)
        else:
            self.batche_ids = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if "loci" in self.data.var.keys():
            chrom, region, strand = self.data.var["loci"][index].split(':')
        else:
            chrom, region = self.data.var.index[index].split(':')
            strand = '.'
        start, end = region.split('-')
        mid = (int(start) + int(end)) // 2
        left, right = mid - self.seq_len//2, mid + self.seq_len//2
        left_pad, right_pad = 0, 0
        if left < 0:
            left_pad = -left_pad
            left = 0
        if right > self.genome[chrom].shape[0]:
            right_pad = right - self.genome[chrom].shape[0]
            right = self.genome[chrom].shape[0]
        seq = self.genome[chrom][left:right]
        if len(seq) < self.seq_len:
            seq = np.concatenate((
                np.full(left_pad, -1, dtype=seq.dtype),
                seq,
                np.full(right_pad, -1, dtype=seq.dtype),
            ))
        if strand == '-':
            seq = get_reverse_strand(seq, integer=True)
        return seq, self.X[index].toarray().flatten().astype(np.float16)
