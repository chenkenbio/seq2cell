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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import dataset
import scanpy as sc
import scbasset
from biock import make_directory, make_logger, get_run_info
from biock.pytorch import model_summary, set_seed
from biock import HG19_FASTA_H5, HG38_FASTA_H5


@torch.no_grad()
@autocast()
def test_model(model, loader, subset=None):
    model.eval()
    all_label = list()
    all_pred = list()
    # all_embedding = list()
    for it, (seq, target) in enumerate(tqdm(loader)):
        seq = seq.to(device)
        # if output_embedding:
            # embedding = model(seq, output_embedding=True).detach()
            # all_embedding.append(embedding.detach().cpu().numpy().astype(np.float16))
        # else:
        output = model(seq)[0].detach()
        output = torch.sigmoid(output).cpu().numpy().astype(np.float16)
        target = target.numpy().astype(np.int8)
        all_pred.append(output)
        all_label.append(target)
    # if output_embedding:
    #     all_embedding = np.concatenate(all_embedding, axis=0)
    #     return all_embedding
    # else:
    all_pred = np.concatenate(all_pred, axis=0) # (n_peaks, n_cells)
    all_label = np.concatenate(all_label, axis=0)
    val_ap = list() # ap per peak
    val_auc = list()
    test_inds = range(all_pred.shape[0])
    for i in tqdm(test_inds, desc="Calculating AP"):
        val_ap.append(average_precision_score(all_label[i], all_pred[i]))
        val_auc.append(roc_auc_score(all_label[i], all_pred[i]))
    val_ap = np.array(val_ap)
    val_auc = np.array(val_auc)
    return val_ap, val_auc


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', "--data", default="/bigdat1/user/chenken/data/single-cell/GSE194122_multiome_neurips2021/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.ATAC.h5ad", help="h5ad data")
    p.add_argument("--watch", choices=("auc", "ap", "mean"), default="mean")
    p.add_argument("-l21", type=float, default=1e-6)
    p.add_argument("-l22", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=0, help="validate every 10 epoches in warmup")
    p.add_argument("--weight-decay", type=float, default=0, help="weight decay")
    p.add_argument("-z", type=int, default=32)
    p.add_argument("-g", choices=("hg19", "hg38", "hg38-shuffle"), required=True)
    p.add_argument("-lr", type=float, default=1e-2)
    p.add_argument('-b', "--batch-size", help="batch size", type=int, default=64)
    p.add_argument("--num-workers", help="number of workers", type=int, default=32)
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--seq-len", type=int, default=1344)
    p.add_argument("-o", "--outdir", help="output directory", default="output")

    p.add_argument("--disable-earlystopping", action="store_true")
    p.add_argument("-w")
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.outdir = make_directory(args.outdir)
    logger = make_logger(title="", filename=os.path.join(args.outdir, "train.log"))
    logger.info(get_run_info(sys.argv, args))

    if args.g == "hg38":
        genome = HG38_FASTA_H5
    elif args.g == "hg38-shuffle":
        genome = "/data0/user/chenken/db/gencode/GRCh38/GRCh38.primary_assembly.shuffled-genome.fa.h5"
    elif args.g == "hg19":
        genome = HG19_FASTA_H5
    ds = dataset.SingleCellDataset(dataset.load_adata(args.data), seq_len=args.seq_len, genome=genome)
    # if ds.X.shape[1] > 1000:
    #     subset = np.random.permutation(np.arange(ds.X.shape[1]))[:1000]
    # else:
    #     subset = None

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4
    )
    
    sampled = np.random.permutation(np.arange(len(ds)))[:1000]
    valid_loader = DataLoader(
        Subset(ds, sampled),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # is_valid = np.isin(ds.chroms, ["chr1"])
    # train_idx = np.arange(len(ds))[~is_valid]
    # valid_idx = np.arange(len(ds))[is_valid]

    # train_loader = DataLoader(
    #     Subset(ds, train_idx),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     prefetch_factor=4
    # )
    
    # valid_loader = DataLoader(
    #     Subset(ds, valid_idx),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = scbasset.scBasset(n_cells=ds.X.shape[1], hidden_size=args.z, seq_len=args.seq_len, batch_ids=ds.batche_ids, bias=not args.no_bias).to(device)
    if args.w is not None:
        model.load_state_dict(torch.load(args.w))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("{}\n{}\n{}\n".format(model, model_summary(model), optimizer))

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.9,
        patience=2,
        min_lr=1e-7
    )

    best_score = 0
    wait = 0
    patience = 20

    train_iters = len(train_loader)

    max_epoch = 1000
    for epoch in range(max_epoch):
        pool = [np.nan for _ in range(10)]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epoch}")
        model.train()
        for it, (seq, target) in enumerate(pbar):
            seq = seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            with autocast():
                output, cell_l2, batch_l2 = model(seq)
                loss = criterion(output, target) + args.l21 * cell_l2 + args.l22 * batch_l2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pool[it % 10] = loss.item()

            lr = optimizer.param_groups[-1]["lr"]
            pbar.set_postfix_str(f"loss/lr={np.nanmean(pool):.4f}/{lr:.3e}")

        del seq, target, loss
        
        if epoch < args.warmup and epoch % 10 != 0:
            logger.info(f"skip validation at epoch {epoch + 1} (warmup)")
            continue
        elif epoch < args.warmup * 2 and epoch % 5 != 0:
            logger.info(f"skip validation at epoch {epoch + 1} (2 * warmup)")
            continue
        elif epoch % 2 != 0:
            logger.info(f"skip validation at epoch {epoch + 1}")
            continue
        

        val_ap, val_auc = test_model(model, valid_loader)

        logger.info("Validation{} AP={:.4f}/{:.4f} AUC={:.4f}/{:.4f}".format((epoch + 1), val_ap.mean(), np.std(val_ap), val_auc.mean(), np.std(val_auc)))

        if args.watch == "auc":
            val_score = val_auc.mean()
        elif args.watch == "ap":
            val_score = val_ap.mean()
        else:
            val_score = (val_auc.mean() + val_ap.mean()) / 2

        scheduler.step(val_score)

        # if args.disable_earlystopping:
        #     continue
        # else:
        if val_score > best_score:
            best_score = val_score
            wait = 0
            torch.save(model.state_dict(), "{}/best_model.pt".format(args.outdir))
            logger.info(f"Epoch {epoch+1}: best model saved\n")
        else:
            wait += 1
            logger.info(f"Epoch {epoch+1}: early stopping patience {wait}/{patience}\n")
            if wait >= patience and not args.disable_earlystopping:
                logger.info(f"Epoch {epoch+1}: early stopping")
                break
    
    # if args.w is None:
    #     model.load_state_dict(torch.load("{}/best_model.pt".format(args.outdir)))
    # else:
    #     model.load_state_dict(torch.load(args.w))

    model.load_state_dict(torch.load("{}/best_model.pt".format(args.outdir)))
    embedding = model.get_embedding().detach().cpu().numpy().astype(np.float32)
    logger.info("embedding: {}".format(embedding.shape))
    adata = sc.AnnData(
        embedding,
        obs=ds.obs,
    )
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    adata.write_h5ad("{}/adata.h5ad".format(args.outdir), compression="gzip")
    from sklearn.metrics import adjusted_rand_score

    logger.info("ARI={}".format(adjusted_rand_score(adata.obs["cell_type"], adata.obs["leiden"])))

