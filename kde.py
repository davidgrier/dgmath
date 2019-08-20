#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import iqr


def akde(x, y, weight=None, alpha=0.5):
    '''Adaptive kernel density estimator

    ...

    Arguments
    ---------
    x: ndarray or list of ndarrays
        Measured data points
    y: ndarray or list of ndarrays
        Points at which the density should be estimated

    Keywords
    --------
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.
    alpha: float
        adaptation factor: alpha >= 0
        0: non-adaptive
        0.5: default
        1: full adaptive (noisy)

    Returns
    -------
    rho: ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values.
    '''
    if isinstance(x, list):
        rho = akde_nd(x, y, weight=weight, alpha=alpha)
    else:
        rho = akde_1d(x, y, weight=weight, alpha=alpha)
    return rho


def kde(x, y, weight=None, scale=None, stats=False):
    '''Kernel density estimator

    ...

    Arguments
    ---------
    x: ndarray or list of ndarrays
        Measured data points
    y: ndarray or list of ndarrays
        Points at which the density should be estimated

    Keywords
    --------
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.
    scale: float
        Scale factor for smoothing.
        Default: Computed from data.
    stats: bool
        Return density, optimal width and variance if set
        Default: False

    Returns
    -------
    rho : ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values.
    '''
    if isinstance(x, list):
        res = kde_nd(x, y, weight=weight, scale=scale, stats=stats)
    else:
        res = kde_1d(x, y, weight=weight, scale=scale, stats=stats)
    return res


def akde_1d(x, y, weight=None, alpha=0.5):
    '''Adaptive kernel density estimator: One dimension

    ...

    Arguments
    ---------
    x: ndarray
        Measured data points
    y: ndarray
        Points at which the density should be estimated

    Keywords
    --------
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.
    alpha: float
        adaptation factor: alpha >= 0
        0: non-adaptive
        0.5: default
        1: full adaptive (noisy)

    Returns
    -------
    rho: ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values.
    '''

    nx = x.size  # number of data points
    ny = y.size  # number of sample points

    # Method described by Silverman Sec. 5.3.1
    # 1. pilot estimate of the density at the data points
    rho, scale, sigma = kde_1d(x, x, stats=True)

    # 2. local bandwidth factor
    g = np.exp(np.mean(np.log(rho)))  # geometric mean density
    factor = (g/rho)**alpha           # Eq. (5.7)
    h = factor * scale                # Scale for each input point

    # 3. adaptive density estimate
    if weight is None:
        weight = 1.
    normalization = weight/(nx * np.sqrt(2.*np.pi) * h)

    rho = np.empty(ny)
    for n in range(ny):
        z = 0.5 * ((x - y[n])/h)**2
        value = normalization * np.exp(-z)
        rho[n] = np.sum(value)
    return rho


def akde_nd(x, y, weight=None, alpha=0.5):
    '''Adaptive kernel density estimator: N dimensions

    ...

    Arguments
    ---------
    x: list of ndarrays
        Measured data points
    y: list of ndarrays
        Points at which the density should be estimated

    Keywords
    --------
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.

    alpha: float
        adaptation factor: alpha >= 0
        0: non-adaptive
        0.5: default
        1: full adaptive (noisy)

    Returns
    -------
    rho: ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values.
    '''
    nd = len(x)     # number of dimensions
    nx = x[0].size  # number of data points
    ny = y[0].size  # number of sample points

    # Method described by Silverman Sec. 5.3.1
    # 1. pilot estimate of the density at the data points
    rho, scale, sigma = kde_nd(x, x, stats=True)

    # 2. local bandwidth factor
    g = np.exp(np.mean(np.log(rho)))  # geometric mean density
    factor = (g/rho)**alpha          # Eq. (5.7)
    h = np.outer(factor, scale)      # Scale for each input point

    # 3. adaptive density estimate
    if weight is None:
        weight = 1.
    normalization = ((weight/nx) *
                     (2.*np.pi * np.sum(h**2, axis=1))**(-nd/2.))

    t = np.array(x).T
    s = np.array(y).T
    rho = np.empty(ny)
    for n in range(ny):
        z = 0.5 * np.sum(((t - s[n, :])/h)**2, axis=1)
        value = normalization * np.exp(-z)
        rho[n] = np.sum(value)
    return rho


def kde_1d(x, y, scale=None, weight=None, stats=False):
    '''Kernel density estimator: One dimension

    ...

    Arguments
    ---------
    x: ndarray
        Measured data points
    y: ndarray
        Points at which the density should be estimated

    Keywords
    --------
    scale: float
        Scale factor for smoothing.
        Default: Computed from data.
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.
    stats: bool
        Return statistical information, if True

    Returns
    -------
    rho: ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values
    scale: ndarray | optional
        Scale factors for each of the dimensions in the list of
        measured data points.
    sigma: ndarray | optional
        Variance of the returned value at each sampling point.
    '''
    nx = x.size  # number of data points
    ny = y.size  # number of sample points

    # optimal smoothing parameter in each dimension
    # Silverman Eqs. (3.30) and (3.31)
    if scale is None:
        sx = x.std()
        rx = iqr(x)
        h = min(sx, rx/1.34) if (rx > 1e-10) else sx
        h *= 0.9/nx**0.2
    else:
        h = scale

    # density estimate
    # Silverman Eq. (2.15) and Table 3.1
    t = x / h
    s = y / h

    if weight is None:
        weight = 1.
    normalization = weight/(nx * np.sqrt(2.*np.pi) * h)

    rho = np.empty(ny)
    if stats:
        sigma = np.empty(ny)
        for n in range(ny):
            z = 0.5 * (t - s[n])**2
            value = normalization * np.exp(-z)
            rho[n] = np.sum(value)
            sigma[n] = np.sum(value**2)
        return rho, h, sigma
    else:
        for n in range(ny):
            z = 0.5 * (t - s[n])**2
            value = normalization * np.exp(-z)
            rho[n] = np.sum(value)
    return rho


def kde_nd(x, y, scale=None, weight=None, stats=False):
    '''Kernel density estimator: N dimensions

    ...

    Arguments
    ---------
    x : list of ndarrays
        Measured data points
    y : list of ndarrays
        Points at which the density should be estimated

    Keywords
    --------
    weight: ndarray
        Value at each of the measured data points.
        Default: 1.
    scale: ndarray
        Scale factor for each dimension of data
        Default: Calculated from data.
    stats: bool
        Return statistical information, if True

    Returns
    -------
    rho: ndarray
        Density at each of the sampling points, optionally weighted
        by the specified values
    scale: ndarray | optional
        Scale factors for each of the dimensions in the list of
        measured data points.
    sigma: ndarray | optional
        Variance of the returned value at each sampling point.
    '''
    nd = len(x)     # number of dimensions
    nx = x[0].size  # number of data points
    ny = y[0].size  # number of sample points

    # optimal smoothing parameter in each dimension
    # Silverman Eqs. (3.30) and (3.31)
    sx = [this.std() for this in x]
    rx = [iqr(this) for this in x]
    h = np.array(sx)
    for n in range(nd):
        if (rx[n] > 1e-10):
            h[n] = min(sx[n], rx[n]/1.34)
    h *= 0.9/nx**0.2

    # density estimate
    # Silverman Eq. (2.15) and Table 3.1
    t = np.array(x).T
    s = np.array(y).T

    if weight is None:
        weight = 1.
    normalization = weight/(nx * np.prod(np.sqrt(2.*np.pi) * h))

    rho = np.empty(ny)
    if stats:
        sigma = np.empty(ny)
        for n in range(ny):
            z = 0.5 * np.sum(((t - s[n, :])/h)**2, axis=1)
            value = np.exp(-z)
            rho[n] = normalization * np.sum(value)
            sigma[n] = normalization**2 * np.sum(value**2)
        return rho, h, sigma
    else:
        for n in range(ny):
            z = 0.5 * np.sum(((t - s[n, :])/h)**2, axis=1)
            rho[n] = normalization * np.sum(np.exp(-z))
    return rho


def distribution(data, value, nx=100, ny=100, adaptive=False):
    x = np.linspace(min(data[0]), max(data[0]), nx)
    y = np.linspace(min(data[1]), max(data[1]), ny)
    xx, yy = np.meshgrid(x, y)
    sample = [xx.ravel(), yy.ravel()]
    if adaptive:
        rhow = akde_nd(data, sample, value)
        rho = akde_nd(data, sample)
    else:
        rhow = kde_nd(data, sample, value)
        rho = kde_nd(data, sample)
    return (rhow/rho).reshape(nx, ny), xx, yy


def example2d():
    import matplotlib.pyplot as plt

    npts = 1000
    sigmax = 1.
    sigmay = 0.5
    mux = 0.
    muy = 0.
    x = np.random.normal(loc=mux, scale=sigmax, size=npts)
    y = np.random.normal(loc=muy, scale=sigmay, size=npts)

    data = [x, y]
    rho, h, sigma = kde(data, data, stats=True)
    sp = plt.scatter(x, y, c=rho, cmap='jet', alpha=0.2)
    cb = plt.colorbar(sp)
    cb.set_label(r'$\rho(x, y)$')
    plt.show()


def example():
    import matplotlib.pyplot as plt

    # Gaussian distributed random points
    mu = 5.     # mean
    sigma = 1.  # variance
    x = np.random.normal(loc=mu, scale=sigma, size=1000)
    y = np.linspace(0, 10)
    plt.scatter(x, 0.*x, marker='+', label='data')

    z = (y - mu)**2 / (2. * sigma**2)
    norm = np.sqrt(2.*np.pi*sigma**2)
    rho = np.exp(-z) / norm
    plt.plot(y, rho, label='Ground Truth')

    rho1, h, sigma = kde(x, y, stats=True)
    plt.plot(y, rho1, label='KDE')

    rho2 = akde(x, y)
    plt.plot(y, rho2, label='AKDE')

    plt.xlabel('Sample value: v')
    plt.ylabel('Probability density: P(v)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example2d()
