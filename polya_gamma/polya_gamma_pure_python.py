"""
polya_gamma.py: code for sampling from a Polya-Gamma distribution PG(b,z).
"""
import numpy as np
import scipy.stats as stats
from typing import List


# ===== ===== Truncated Inverse Gaussian IG(mu, lambda) * 1([0,t]) Sampler
def _tinvgauss_rv_mu_lte_t(mu, t):
    while True:
        X = stats.invgauss.rvs(mu=mu)
        if (X < t):
            break
    return X


def _tinvgauss_rv_mu_gt_t(mu, t):
    z = 1.0 / mu
    while True:
        while True:
            e1 = stats.expon.rvs(scale=1.0)
            e2 = stats.expon.rvs(scale=1.0)
            if e1*e1 <= 2*e2 / t:
                break
        X = t / (1 + t*e1)**2
        alpha = np.exp(-0.5 * z*z * X)
        U = np.random.rand()
        if U <= alpha:
            break
    return X


def tinvgauss_rv(mu, t): 
    """
    Return a single sample X ~ IG(mu, 1.0) * 1([0,t])
    """
    assert (mu > 0.0), "IG(mu, 1,0) * 1([0,t]) only admits positive mu"
    assert (t > 0.0), "IG(mu, 1.0) * 1([0,t]) only admits positive t"
    if mu > t:
        return _tinvgauss_rv_mu_gt_t(mu, t)
    else:
        return _tinvgauss_rv_mu_lte_t(mu, t)


def tinvgauss_mu_infty_rv(t):
    """
    Return a single sample X ~ IG(mu=INFTY, 1.0) * 1([0,t]).
    """
    while True:
        e1 = stats.expon.rvs(scale=1.0)
        e2 = stats.expon.rvs(scale=1.0)
        if e1*e1 <= 2*e2 / t:
            break
    X = t / (1 + t*e1)**2
    return X


# ===== ===== PG(1,z) Sampler
def piecewise_coeff(n, x, t):
    """
    Compute the piecewise coefficients for the alternating-series representation of the
    density of the Jacobi theta distribution.
    
    n : int >= 0
    x : float > 0
    t: float > 0
    """
    assert (n >= 0), "a_n(x|t) only admits non-negative integer `n` parameter"
    assert (x > 0.0), "a_n(x|t) only admits positive float `x` parameter"
    assert (t > 0.0), "a_n(x|t) only admits non-negative float `t` parameter"
    out = np.pi * (n + 0.5)
    if x <= t:
        out *= (2.0 / (np.pi * x))**(1.5)
        out *= np.exp( (-2.0 * (n+0.5)**2.) / x )
    else:
        out *= np.exp( -0.5 * np.pi**2 * (n+0.5)**2 * x )
    return out


def tilted_jacobi_theta_rv(z):
    """
    Return a single sample X ~ J*(1,z), the tilted Jacobi theta distribution.
    """
    assert (z > 0.0), "J*(1,z) only admits positive float `z` parameter"
    # --- fixed parameters
    t = 0.64 # empirically-discovered optimal cutoff point for t; cf. Devroye 2009.
    K = (np.pi**2 / 8.0) + (z**2 / 2.0)
    p = (np.pi / (2.0*K)) * np.exp(-K*t)
    q = 2.0 * np.exp(-np.abs(z)) * stats.invgauss.cdf(t, mu=(1.0/z), scale=1.0)
    # --- accept-reject sampler:
    while True:
        U = np.random.rand()
        V = np.random.rand()
        # sample proposal for X:
        if U < (p / (p+q)):
            X = t + stats.expon.rvs(scale=1.0) / K
        else:
            X = tinvgauss_rv(1.0/z, t)
        # compute partial sums and check for acceptance criterion:
        S = piecewise_coeff(0, X, t)
        Y = V * S
        n = 0
        while True:
            n += 1
            if (n % 2 == 1):
                # n is odd
                S -= piecewise_coeff(n,X,t)
                if Y < S:
                    return X
                else:
                    continue
            else:
                # n is even
                S += piecewise_coeff(n,X,t)
                if Y > S:
                    break
                else:
                    continue


def tilted_jacobi_theta_z0_rv():
    """
    Return a single sample X ~ J*(1,0.0).
    """
    # --- fixed parameters
    t = 0.64 # empirically-discovered optimal cutoff point for t; cf. Devroye 2009.
    K = (np.pi**2 / 8.0)
    p = (np.pi / (2.0*K)) * np.exp(-K*t)
    q = 4.0 * stats.norm.cdf(-np.sqrt(1.0/t))
    # --- accept-reject sampler:
    while True:
        U = np.random.rand()
        V = np.random.rand()
        # sample proposal for X:
        if U < (p / (p+q)):
            X = t + stats.expon.rvs(scale=1.0) / K
        else:
            X = tinvgauss_mu_infty_rv(t)
        # compute partial sums and check for acceptance criterion:
        S = piecewise_coeff(0, X, t)
        Y = V * S
        n = 0
        while True:
            n += 1
            if (n % 2 == 1):
                # n is odd
                S -= piecewise_coeff(n,X,t)
                if Y < S:
                    return X
                else:
                    continue
            else:
                # n is even
                S += piecewise_coeff(n,X,t)
                if Y > S:
                    break
                else:
                    continue
                

# ===== ===== PG(b,z) Sampler
def polya_gamma_rv(b, z):
    """
    Return a single sample X ~ PG(b,z). Relies on the fact that sampling
      X ~ PG(1,z)
    is equivalent to sampling
      X ~ J*(1, z/2.0) / 4.0
    and that 
      X ~ PG(b,z)
    is equivalent to
      X == sum([ X ~ PG(1,z) for _ in range(b) ]).

    Caveats:
    * `b` must be positive integer
    * random variates generated sequentially, i.e. no `size` parameter
    """
    assert (b > 0), "PG(b,z) only admits positive integer `b` parameter"
    if abs(z) > 1e-12: # (anything smaller than 1e-12 is considered zero)
        return sum([ tilted_jacobi_theta_rv(abs(z) / 2.0) / 4.0 for _ in range(b) ])
    else:
        return sum([ tilted_jacobi_theta_z0_rv() / 4.0 for _ in range(b) ])
