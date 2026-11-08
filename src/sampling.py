"""
Noise / direction strategies for the re-sampling step

    w' = w + coef * resampling_coef * noise(w),    coef = (cos(student, teacher) + 1) / 2

`isotropic` is the original SynthDistill rule (noise ~ N(0, I)). New strategies are
registered with @register and receive (w, ctx) where w is (N, w_dim) and ctx holds
shared state (device, w_dim, W statistics, ...). Planned additions: PCA (GANSpace)
directions, attribute directions (pose, age), adversarial directions.
"""
import numpy as np
import torch

STRATEGIES = {}


def register(name):
    def deco(cls):
        STRATEGIES[name] = cls
        return cls
    return deco


class Strategy:
    needs_w_stats = False

    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, w):
        raise NotImplementedError


@register('isotropic')
class Isotropic(Strategy):
    """Original rule: N(0, I) in W. Note: W itself only has std ~0.41 per dimension."""

    def __call__(self, w):
        # identical call to the original code -> identical random stream
        return torch.randn(w.shape[0], self.ctx['w_dim'], device=self.ctx['device'])


@register('wcov')
class WCovariance(Strategy):
    """N(0, Sigma_W): noise shaped like the distribution of W (stays closer to the generator's manifold)."""
    needs_w_stats = True

    def __init__(self, ctx):
        super().__init__(ctx)
        cov = torch.as_tensor(ctx['w_cov'], dtype=torch.float64)
        cov = cov + 1e-6 * torch.eye(cov.shape[0], dtype=torch.float64)
        self.L = torch.linalg.cholesky(cov).float().to(ctx['device'])

    def __call__(self, w):
        return torch.randn(w.shape[0], self.ctx['w_dim'], device=self.ctx['device']) @ self.L.T


def make_strategy(name, ctx):
    if name not in STRATEGIES:
        raise ValueError(f'unknown noise strategy {name!r}; available: {sorted(STRATEGIES)}')
    return STRATEGIES[name](ctx)


def load_w_stats(path):
    d = np.load(path)
    return d['mean'], d['cov']
