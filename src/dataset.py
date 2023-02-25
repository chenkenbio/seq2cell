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


def load_adata(data) -> AnnData:
    return sc.read_h5ad(data)

ONEHOT = np.concatenate((
    np.zeros((1, 4), dtype=np.float16),
    np.eye(4, dtype=np.float16),
))

class SingleCellDataset(Dataset):
    def __init__(self, data: AnnData, seq_len=1344, genome=HG38_FASTA_H5):
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))
        self.data = data
        self.seq_len = seq_len
        self.genome = h5py.File(genome, 'r')
        batch2index = {batch:index for index, batch in enumerate(self.data.obs['batch'].unique())}
        self.batches = np.asarray(self.data.obs['batch'].map(batch2index).values)
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T)
        del self.data.X
        if "chrom" in self.var.keys():
            self.chroms = self.var["chrom"]

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
        seq = self.genome[chrom][mid - self.seq_len//2:mid + self.seq_len//2]
        if strand == '-':
            seq = get_reverse_strand(seq, integer=True)
        # seq = ONEHOT[seq].T
        return seq, self.X[index].toarray().flatten().astype(np.float16), index, self.batches
