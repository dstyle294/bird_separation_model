import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from asteroid.data.wsj0_mix import Wsj0mixDataset


class Wsj0mixVariable(Dataset):
    """Variable-source WSJ0-style dataset wrapper.

    This wraps several fixed-source Wsj0mixDataset instances and returns a
    mixed batch that may contain different numbers of sources.
    """

    def __init__(
        self,
        json_dirs: List[str],
        n_srcs: List[int],
        sample_rate: int = 8000,
        seglen: float = 4.0,
        minlen: float = 2.0,
    ):
        super().__init__()
        assert len(json_dirs) == len(n_srcs), "json_dirs and n_srcs must have the same length"
        self.datasets = []
        self.n_srcs = list(n_srcs)
        self.dataset_sizes = []

        for json_dir, n_src in zip(json_dirs, n_srcs):
            dataset = Wsj0mixDataset(json_dir, n_src=n_src, sample_rate=sample_rate, segment=seglen)
            self.datasets.append(dataset)
            self.dataset_sizes.append(len(dataset))

        if len(self.dataset_sizes) == 0:
            raise ValueError("No datasets were created for Wsj0mixVariable")

        self.cum_sizes = np.cumsum(self.dataset_sizes)
        self.total_size = int(self.cum_sizes[-1])

    def __len__(self):
        return self.total_size

    def __getitem__(self, index: int):
        dataset_index = int(np.searchsorted(self.cum_sizes, index, side="right"))
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - int(self.cum_sizes[dataset_index - 1])

        mixture, sources = self.datasets[dataset_index][sample_index]
        num_sources = self.n_srcs[dataset_index]
        return mixture, sources, num_sources


def _collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    mixtures, sources, num_sources = zip(*batch)
    batch_size = len(mixtures)
    ilens = [m.shape[-1] for m in mixtures]
    max_len = max(ilens)
    max_src = max(num_sources)

    mixture_tensor = torch.zeros(batch_size, max_len, dtype=mixtures[0].dtype)
    source_tensor = torch.zeros(batch_size, max_src, max_len, dtype=sources[0].dtype)
    num_sources_tensor = torch.tensor(num_sources, dtype=torch.long)
    ilens_tensor = torch.tensor(ilens, dtype=torch.long)

    for i, (m, s) in enumerate(zip(mixtures, sources)):
        mixture_tensor[i, : m.shape[-1]] = m
        source_tensor[i, : s.shape[0], : s.shape[1]] = s

    return mixture_tensor, source_tensor, ilens_tensor, num_sources_tensor
