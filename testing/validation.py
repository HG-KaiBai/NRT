"""Reproducible 80/20 splits within the data supplied to a neural fit."""

import torch


def training_validation_split(n_rows, *, seed=None, device="cpu"):
    if n_rows < 2:
        raise ValueError("at least two rows are required for training and validation")
    generator = torch.Generator().manual_seed(torch.initial_seed() if seed is None else seed)
    order = torch.randperm(n_rows, generator=generator).to(device)
    n_train = int(0.8 * n_rows)
    return order[:n_train], order[n_train:]
