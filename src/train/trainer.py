'''PyTorch training loop for the ST-GNN forecasters.

Responsibilities
  * seed all RNGs for reproducibility (design doc §8.3, §12: same seeds/splits);
  * mini-batch training with Adam + gradient clipping;
  * point loss (Huber) or quantile loss (pinball) depending on the model head;
  * early stopping on a chronological validation split;
  * device handling (CPU/CUDA);
  * predict returning point or quantile forecasts in MODEL space.

Everything here is fold-agnostic: it receives already-windowed, already-scaled
tensors from src/train/dataset.py, so it cannot leak. Torch is imported lazily
inside functions so importing this module never requires torch.
'''

from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer_cuda: bool = True) -> str:
    import torch
    return 'cuda' if (prefer_cuda and torch.cuda.is_available()) else 'cpu'


def _masked_huber(pred, target, delta=1.0):
    import torch
    import torch.nn.functional as F
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=delta)


def _masked_pinball(pred_q, target, quantiles):
    '''pred_q: (B,N,H,Q); target: (B,N,H); quantiles: 1-D tensor (Q,).'''
    import torch
    t = target.unsqueeze(-1)                          # (B,N,H,1)
    mask = torch.isfinite(t)
    q = quantiles.view(1, 1, 1, -1)
    diff = torch.where(mask, t - pred_q, torch.zeros_like(pred_q))
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    denom = mask.expand_as(loss).sum().clamp(min=1)
    return loss.sum() / denom


class TrainConfig:
    def __init__(self, epochs=100, batch_size=16, lr=1e-3, weight_decay=1e-4,
                 patience=12, grad_clip=5.0, huber_delta=1.0, seed=0,
                 device=None, verbose=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.grad_clip = grad_clip
        self.huber_delta = huber_delta
        self.seed = seed
        self.device = device
        self.verbose = verbose


def train_model(model, train_samples: dict, val_samples: dict | None,
                cfg: TrainConfig, quantiles: list | None = None):
    '''Train model in place; return (model, history_dict).

    train_samples / val_samples : dicts from src.train.dataset.make_samples,
        with keys 'X' (S,N,L,C) and 'y' (S,N,H), in MODEL space.
    quantiles : list of quantile levels if the model has a QuantileHead,
        else None (point model).
    '''
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    set_seed(cfg.seed)
    device = cfg.device or pick_device()
    model = model.to(device)

    Xtr = torch.from_numpy(train_samples['X']).float()
    Ytr = torch.from_numpy(train_samples['y']).float()
    if Xtr.shape[0] == 0:
        raise ValueError('no training samples; fold too short for the window.')
    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    has_val = val_samples is not None and val_samples['X'].shape[0] > 0
    if has_val:
        Xva = torch.from_numpy(val_samples['X']).float().to(device)
        Yva = torch.from_numpy(val_samples['y']).float().to(device)

    qt = torch.tensor(quantiles, dtype=torch.float32, device=device) if quantiles else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def loss_fn(pred, target):
        if qt is not None:
            return _masked_pinball(pred, target, qt)
        return _masked_huber(pred, target, cfg.huber_delta)

    best_val, best_state, bad = np.inf, None, 0
    hist = {'train_loss': [], 'val_loss': []}

    for epoch in range(cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.item()); nb += 1
        train_loss = tot / max(nb, 1)
        hist['train_loss'].append(train_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(Xva), Yva).item())
            hist['val_loss'].append(vloss)
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if cfg.verbose:
                print(f'  epoch {epoch:3d}  train={train_loss:.4f}  val={vloss:.4f}  bad={bad}')
            if bad >= cfg.patience:
                break
        elif cfg.verbose:
            print(f'  epoch {epoch:3d}  train={train_loss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def predict(model, samples: dict, device: str | None = None,
            batch_size: int = 64) -> np.ndarray:
    '''Return model-space predictions for samples.

    Output shape: (S, N, H) for point models, (S, N, H, Q) for quantile models.
    '''
    import torch
    device = device or pick_device()
    model = model.to(device).eval()
    X = torch.from_numpy(samples['X']).float()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].to(device)
            outs.append(model(xb).cpu().numpy())
    if not outs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
