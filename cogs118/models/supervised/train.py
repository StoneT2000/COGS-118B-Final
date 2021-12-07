import os.path as osp
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
def evaluate(model, val_dl: DataLoader, verbose=True, nb=False):
    model.eval()
    _tqdm = tqdm_nb if nb else tqdm
    if verbose:
        pbar = _tqdm(enumerate(val_dl), total=len(val_dl), position=0, leave=True)
        pbar.set_description("Evaluation Progress")
    with torch.no_grad():
        batch_size = val_dl.batch_size
        avg_loss = defaultdict(int)
        num_train_imgs = len(val_dl.dataset)
        c = 0
        for batch_idx, imgs in enumerate(val_dl):
            c += 1

            res = model(imgs)

            loss = model.loss_function(
                *res,
                M_N=batch_size / num_train_imgs,  # weighting for each batch...
                batch_idx=batch_idx,
            )
            for k, v in loss.items():
                avg_loss[k] += v.item()
            if verbose:
                pbar.update()
        for k in avg_loss.keys():
            avg_loss[k] /= c
    model.train()
    if verbose:
        pbar.close()
    return avg_loss
def train(
    model,
    optim: torch.optim.Optimizer,
    epochs: int,
    train_dl: DataLoader,
    val_dl: DataLoader,
    start_epoch=0,
    save_freq=10,
    save_best=True,
    save_dir="./",
    prev_best_loss=np.inf,
    verbose=True,
    nb=False,
    train_cb=None,
):
    """
    train the vae model
    """
    _tqdm = tqdm_nb if nb else tqdm
    model.train()
    optimizer_idx = 0
    num_train_imgs = len(train_dl.dataset)
    batch_size = train_dl.batch_size
    if verbose:
        epoch_pbar = _tqdm(range(epochs), position=0, leave=True)
        epoch_pbar.set_description("Progress")
        pbar = _tqdm(enumerate(train_dl), total=len(train_dl), position=0, leave=True)
        pbar.set_description("Current Epoch Progress")
        epoch_pbar.update(start_epoch)
    for epoch in range(start_epoch, start_epoch + epochs):
        avg_loss = defaultdict(int)
        c = 0
        if verbose:
            epoch_pbar.update()
            pbar.reset()
        for batch_idx, imgs in enumerate(train_dl):

            c += 1
            # imgs (B, C=1, W, H)
            optim.zero_grad()
            res = model(imgs)

            loss = model.loss_function(
                *res,
                M_N=batch_size / num_train_imgs,  # weighting for each batch...
                optimizer_idx=optimizer_idx,
                batch_idx=batch_idx,
            )
            for k, v in loss.items():
                avg_loss[k] += v.item()
            optimizer_idx += 1
            loss["loss"].backward()
            optim.step()
            if verbose:
                pbar.update()
        for k in avg_loss.keys():
            avg_loss[k] /= c
        eval_loss = evaluate(model, val_dl, nb=nb, verbose=False)
        if train_cb:
            train_cb(epoch=epoch, loss=avg_loss, eval_loss=eval_loss, model=model)
        save = False
        if eval_loss["loss"] < prev_best_loss:
            prev_best_loss = eval_loss["loss"]
            if save_best:
                save = True
        if epoch % save_freq == 0 or epoch == epochs - 1:
            save = True
        if save:
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optim_State_dict=optim.state_dict(),
                    epoch=epoch,
                    loss=avg_loss,
                    eval_loss=eval_loss,
                    prev_best_loss=prev_best_loss,
                ),
                osp.join(save_dir, f"ckpt_{epoch}.pt"),
            )