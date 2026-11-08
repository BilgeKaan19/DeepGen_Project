"""StyleGAN loading and the face synthesis pipeline shared by train.py and generate_dataset.py."""
import os
import pickle
import sys

import torch

from src.Crop import Crop_and_resize


def load_stylegan(stylegan_repo, stylegan_pkl, device, ref_ops=False):
    """Returns the G_ema network with mapping/synthesis in eval mode on `device`."""
    if not os.path.isdir(os.path.join(stylegan_repo, 'torch_utils')):
        sys.exit(f'StyleGAN repo not found at {stylegan_repo}. Run: git clone https://github.com/NVlabs/stylegan3')
    if not os.path.isfile(stylegan_pkl):
        sys.exit(f'StyleGAN weights not found at {stylegan_pkl}. See README for the download link.')
    if stylegan_repo not in sys.path:
        sys.path.append(stylegan_repo)
    import torch_utils  # noqa: F401  (needed to unpickle the generator)
    if ref_ops:
        # pure-PyTorch reference ops instead of JIT-compiled CUDA kernels (same math, slower)
        from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu
        for op in (bias_act, upfirdn2d, filtered_lrelu):
            op._init = lambda: False
        print('************ NOTE: StyleGAN uses reference (non-CUDA-kernel) ops')
    with open(stylegan_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema']
    G.synthesis.eval().to(device)
    G.mapping.eval().to(device)
    return G


@torch.no_grad()
def synthesize(synthesis, w):
    """w: (N, num_ws, w_dim) -> aligned 112x112 RGB faces in (0, 1), exactly as in the original train.py."""
    img = synthesis(w).detach()
    img = torch.clamp(img, min=-1, max=1)
    img = (img + 1) / 2.0  # range: (0,1)
    return Crop_and_resize(img)


def to_uint8(img):
    """(0,1) float images -> uint8 (the precision images are stored with in a cached dataset)."""
    return (img * 255.).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def compute_w_stats(mapping, z_dim, device, n=100_000, seed=0, batch=10_000):
    """Mean and covariance of W = mapping(z), z ~ N(0, I), truncation_psi = 1 (own RNG, global RNG untouched)."""
    g = torch.Generator(device=device).manual_seed(seed)
    ws = []
    for s in range(0, n, batch):
        z = torch.randn(min(batch, n - s), z_dim, device=device, generator=g)
        ws.append(mapping(z=z, c=None, truncation_psi=1.0)[:, 0].double())
    W = torch.cat(ws)
    mean = W.mean(0)
    cov = torch.cov(W.T)
    return mean.cpu().numpy(), cov.cpu().numpy()
