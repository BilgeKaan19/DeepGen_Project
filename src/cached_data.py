"""Reading a dataset written by generate_dataset.py (memory-mapped, shuffled per epoch)."""
import json
import os

import numpy as np
import torch


class CachedDataset:
    def __init__(self, root):
        self.root = root
        meta_path = os.path.join(root, 'meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f'{meta_path} not found -- create the dataset with generate_dataset.py')
        with open(meta_path) as f:
            self.meta = json.load(f)
        if not self.meta.get('complete'):
            raise RuntimeError(f'{root} is incomplete ({self.meta["done"]}/{self.meta["n"]}); '
                               'finish it by re-running generate_dataset.py with the same options')
        self.n = self.meta['n']
        self.w_stats_path = os.path.join(root, 'w_stats.npz')

    def loader(self, batch_size, epoch, seed, num_workers=8):
        """One pass over the data in a random order (fixed by seed and epoch); last partial batch dropped."""
        perm = np.random.RandomState((seed * 1_000_003 + epoch) % 2**32).permutation(self.n)
        ds = _EpochBatches(self.root, perm, batch_size)
        return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers,
                                           pin_memory=True, prefetch_factor=4 if num_workers else 2)

    def batches_per_epoch(self, batch_size):
        return self.n // batch_size


class _EpochBatches(torch.utils.data.Dataset):
    def __init__(self, root, perm, batch_size):
        self.root, self.perm, self.B = root, perm, batch_size
        self._mm = None

    def __len__(self):
        return len(self.perm) // self.B

    def _open(self):
        if self._mm is None:  # opened lazily in each worker process
            self._mm = [np.load(os.path.join(self.root, f'{k}.npy'), mmap_mode='r')
                        for k in ('w', 'images', 'teacher_emb')]
        return self._mm

    def __getitem__(self, i):
        W, IMG, EMB = self._open()
        idx = np.sort(self.perm[i * self.B:(i + 1) * self.B])   # sorted -> sequential-ish disk reads
        return (torch.from_numpy(np.ascontiguousarray(W[idx])),
                torch.from_numpy(np.ascontiguousarray(IMG[idx])),
                torch.from_numpy(np.ascontiguousarray(EMB[idx])))
