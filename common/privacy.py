"""(epsilon, delta) helpers for the optional privacy mode.

Mechanism (client-level, adapted to the logits exchange):

* each client clips its public-set probability vector to L2 norm <= clip_norm
  (C) before sending it
* the server adds Gaussian noise N(0, sigma^2 * s^2) to the averaged
  probabilities (the consensus), where s = C * (largest client share of the
  round) is the L2 sensitivity of the weighted mean w.r.t. one client
* accounting: the analytic Gaussian mechanism (Balle and Wang 2018) per
  round, composed over rounds with basic composition

This is a valid but conservative bound (a moments/RDP accountant would give
tighter numbers). Scope, stated honestly: it protects the released per-round
consensus (what every participant receives); it does not hide individual
updates from the server (there is no secure aggregation).
"""
from __future__ import annotations

import math

import numpy as np


def l2_norm_array(arr: np.ndarray) -> float:
    """Global L2 norm of an array (flattened)."""
    return float(np.sqrt(np.sum(np.asarray(arr, dtype=np.float64) ** 2)))


def clip_array(arr: np.ndarray, clip_norm: float) -> tuple:
    """Scale an array so its L2 norm is at most `clip_norm`.

    Returns (clipped_array, was_clipped, original_norm). The original array is
    returned unchanged when it already fits.
    """
    arr = np.asarray(arr, dtype=np.float32)
    norm = l2_norm_array(arr)
    if norm <= clip_norm or norm == 0.0:
        return arr, False, norm
    return (arr * (clip_norm / norm)).astype(np.float32), True, norm


def add_gaussian_noise_array(arr: np.ndarray, sigma: float, sensitivity: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Add N(0, (sigma * sensitivity)^2) noise to every element."""
    std = sigma * sensitivity
    arr32 = np.asarray(arr, dtype=np.float32)
    noise = rng.normal(0.0, std, size=arr32.shape).astype(np.float32)
    return (arr32 + noise).astype(np.float32)


def l2_norm_dict(arrays: dict) -> float:
    """Global L2 norm of a dict of arrays (all tensors flattened)."""
    squared = sum(float(np.sum(np.asarray(arr, dtype=np.float64) ** 2)) for arr in arrays.values())
    return math.sqrt(squared)


def clip_dict(arrays: dict, clip_norm: float) -> tuple:
    """Scale every array in a dict so the global L2 norm is at most clip_norm.

    Returns (clipped_dict, was_clipped, original_norm).
    """
    norm = l2_norm_dict(arrays)
    if norm <= clip_norm or norm == 0.0:
        return arrays, False, norm
    scale = clip_norm / norm
    clipped = {key: (np.asarray(arr) * scale).astype(np.float32) for key, arr in arrays.items()}
    return clipped, True, norm


def add_gaussian_noise_dict(arrays: dict, sigma: float, sensitivity: float,
                            rng: np.random.Generator) -> dict:
    """Add N(0, (sigma * sensitivity)^2) noise to every element of a dict."""
    std = sigma * sensitivity
    out = {}
    for key, arr in arrays.items():
        arr32 = np.asarray(arr, dtype=np.float32)
        noise = rng.normal(0.0, std, size=arr32.shape).astype(np.float32)
        out[key] = (arr32 + noise).astype(np.float32)
    return out


def mean_sensitivity(clip_norm: float, sample_counts: list) -> float:
    """L2 sensitivity of the sample-weighted mean of clipped vectors w.r.t.
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
