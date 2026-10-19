"""(epsilon, delta) differential privacy helpers, DP-FedAvg style.

Mechanism (following Andrew et al. 2021, "Differentially Private Federated
Learning with Client-Level DP", simplified):

* every client clips its update delta to L2 norm <= clip_norm C
* the server averages the clipped deltas and adds Gaussian noise
  N(0, sigma^2 * s^2), where s = C * (largest client share of the round)
  is the L2 sensitivity of the weighted mean with respect to one client
* accounting: the analytic Gaussian mechanism (Balle and Wang 2018) per
  round, composed over rounds with basic composition

This is a valid but conservative bound (a moments/RDP accountant would give
tighter numbers). Scope, stated honestly: protects the released per-round
aggregate against inferences about a single client's participation/update.
It does NOT hide individual updates from the server (the server computes the
noisy mean and sees each clipped delta; no secure aggregation here), and it
does not protect the raw local data beyond what the clipping bounds allow.
"""
from __future__ import annotations

import math

import numpy as np


def l2_norm(delta: dict) -> float:
    """Global L2 norm of a delta dict (all tensors flattened)."""
    squared = sum(float(np.sum(np.asarray(arr, dtype=np.float64) ** 2)) for arr in delta.values())
    return math.sqrt(squared)


def clip_delta(delta: dict, clip_norm: float) -> tuple:
    """Scale a delta dict so its global L2 norm is at most `clip_norm`.

    Returns (delta_clipped, was_clipped, original_norm). The original dict is
    returned unchanged when it already fits.
    """
    norm = l2_norm(delta)
    if norm <= clip_norm or norm == 0.0:
        return delta, False, norm
    scale = clip_norm / norm
    clipped = {k: (np.asarray(arr) * scale).astype(np.float32) for k, arr in delta.items()}
    return clipped, True, norm


def add_gaussian_noise(delta: dict, sigma: float, sensitivity: float, rng: np.random.Generator) -> dict:
    """Add N(0, (sigma * sensitivity)^2) noise to every element of the delta."""
    std = sigma * sensitivity
    out = {}
    for k, arr in delta.items():
        arr32 = np.asarray(arr, dtype=np.float32)
        noise = rng.normal(0.0, std, size=arr32.shape).astype(np.float32)
        out[k] = (arr32 + noise).astype(np.float32)
    return out


def mean_sensitivity(clip_norm: float, sample_counts: list) -> float:
    """L2 sensitivity of the sample-weighted mean of clipped deltas w.r.t.
    one client's replacement:  clip_norm * max(n_k) / sum(n_k)."""
    total = float(sum(sample_counts))
    if total <= 0:
        return float(clip_norm)
    return float(clip_norm) * (max(sample_counts) / total)


def _delta_of_gaussian(sigma: float, eps: float) -> float:
    """delta(eps) for a Gaussian mechanism with L2 sensitivity 1 and noise
    multiplier sigma (analytic form, Balle and Wang 2018)."""
    def phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    a = 1.0 / (2.0 * sigma)
    return phi(a - eps * sigma) - math.exp(eps) * phi(-a - eps * sigma)


def epsilon_gaussian_analytic(sigma: float, delta_round: float) -> float:
    """Smallest eps such that the Gaussian mechanism (sensitivity 1) with
    noise multiplier sigma is (eps, delta_round)-DP."""
    if sigma <= 0:
        return math.inf
    lo, hi = 1e-9, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _delta_of_gaussian(sigma, mid) > delta_round:
            lo = mid
        else:
            hi = mid
    return hi


def total_epsilon(sigma: float, rounds: int, target_delta: float) -> float:
    """(eps, delta) accounting over `rounds` rounds via basic composition.

    Splits the delta budget evenly across rounds (delta_round = target_delta /
    rounds) and composes epsilon additively. Conservative, valid, and easy to
    re-derive on the slides.
    """
    if sigma <= 0:
        return math.inf
    rounds = max(int(rounds), 1)
    delta_round = float(target_delta) / rounds
    return rounds * epsilon_gaussian_analytic(sigma, delta_round)
