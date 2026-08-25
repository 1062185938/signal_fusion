"""Dataset-to-DataLoader splitting policies used by offline training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_training_loaders(
    X: np.ndarray,
    Y: np.ndarray,
    meta: dict[str, Any],
    *,
    split_mode: str,
    max_samples: int | None,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    """Build the historical random or burst-grouped train/val/test split."""

    if split_mode == "group" and max_samples is not None and max_samples > 0:
        raise ValueError("max_samples is not supported when split_mode=group.")

    kwargs = {"num_workers": num_workers}
    if torch.cuda.is_available():
        kwargs["pin_memory"] = True

    if split_mode == "random":
        if max_samples is not None and max_samples > 0 and max_samples < len(Y):
            rng = np.random.default_rng(seed)
            sample_indices = rng.choice(len(Y), size=max_samples, replace=False)
            X = X[sample_indices]
            Y = Y[sample_indices]
            print(f"Smoke test mode: using max_samples={max_samples}")

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        total_len = len(dataset)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len

        train_set, val_set, test_set = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True, **kwargs
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, **kwargs
        )

        print("Split mode: random")
        print(f"Total samples: {total_len}")
        print(f"Train samples: {train_len}")
        print(f"Val samples: {val_len}")
        print(f"Test samples: {test_len}")
    elif split_mode == "group":
        if "burst_id" not in meta:
            raise ValueError("split_mode=group requires burst_id in dataset metadata.")

        burst_id = np.asarray(meta["burst_id"]).squeeze()
        if burst_id.shape[0] != X.shape[0] or burst_id.shape[0] != len(Y):
            raise ValueError(
                "burst_id length must match X.shape[0] and y length for "
                "split_mode=group."
            )

        unique_burst_ids = np.unique(burst_id)
        shuffled_bursts = unique_burst_ids.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled_bursts)

        total_bursts = len(shuffled_bursts)
        train_burst_len = int(0.7 * total_bursts)
        val_burst_len = int(0.15 * total_bursts)

        train_bursts = shuffled_bursts[:train_burst_len]
        val_bursts = shuffled_bursts[train_burst_len : train_burst_len + val_burst_len]
        test_bursts = shuffled_bursts[train_burst_len + val_burst_len :]

        train_burst_set = set(train_bursts.tolist())
        val_burst_set = set(val_bursts.tolist())
        test_burst_set = set(test_bursts.tolist())
        if (
            train_burst_set.intersection(val_burst_set)
            or train_burst_set.intersection(test_burst_set)
            or val_burst_set.intersection(test_burst_set)
        ):
            raise RuntimeError("Group split failed: burst_id overlap detected.")

        train_mask = np.isin(burst_id, train_bursts)
        val_mask = np.isin(burst_id, val_bursts)
        test_mask = np.isin(burst_id, test_bursts)

        train_set = TensorDataset(
            torch.from_numpy(X[train_mask]), torch.from_numpy(Y[train_mask])
        )
        val_set = TensorDataset(
            torch.from_numpy(X[val_mask]), torch.from_numpy(Y[val_mask])
        )
        test_set = TensorDataset(
            torch.from_numpy(X[test_mask]), torch.from_numpy(Y[test_mask])
        )

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, **kwargs
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, **kwargs
        )

        print("Split mode: group")
        print(f"Total bursts: {total_bursts}")
        print(f"Train bursts: {len(train_bursts)}")
        print(f"Val bursts: {len(val_bursts)}")
        print(f"Test bursts: {len(test_bursts)}")
        print(f"Train samples: {len(train_set)}")
        print(f"Val samples: {len(val_set)}")
        print(f"Test samples: {len(test_set)}")
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    return train_loader, val_loader, test_loader


__all__ = ["build_training_loaders"]
