# -*- coding: utf-8 -*-
"""
Probability Density Functions

A collection of probability density functions
"""
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import splev, splrep
from scipy.stats import gaussian_kde

from .base import BaseDistribution


class HistogramDistribution(BaseDistribution):
    """Use a histogram to estimate a probability density function."""

    def __init__(self, nbins: int = 100):
        """
        Parameters
        ----------
        nbins : int, optional
            The number of bins to use in the histogram (default is 100)
        """
        super().__init__()

        # Set the number of bins
        self.nbins = nbins

    def fit(self, data: Iterable[float]) -> 'HistogramDistribution':
        """
        Parameters
        ----------
        data : array_like
            The data to fit

        Returns
        -------
        HistogramDistribution
            The fitted histogram distribution
        """
        # Fit the histogram distribution
        self._hist, self._bins = np.histogram(data, bins=self.nbins, density=True)

        # Set the fitted status
        self.fitted = True

        return self

    def __call__(self, x: float) -> float:
        """
        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point

        Raises
        ------
        RuntimeError
            If the histogram distribution has not been fitted
        """
        if not self.fitted:
            raise RuntimeError(
                'The histogram distribution must be fitted before it can be evaluated. Call self.fit(data) to fit the histogram distribution.'
            )

        if (x < self._bins[0]) or (x > self._bins[-2]):
            return 0

        # Find the bin that contains the point
        idx = np.digitize(x, self._bins)

        return self._hist[idx]

    def integrate(self,
                  a: float = -np.inf,
                  b: float = np.inf,
                  moment: float = 1) -> Tuple[float, float]:
        """
        Integrate the probability density function from a to b.

        Parameters
        ----------
        a : float, optional
            The lower bound of the integration (default is -inf)
        b : float, optional
            The upper bound of the integration (default is inf)
        moment : float, optional
            The moment to compute (default is 0)

        Returns
        -------
        Tuple[float, float]
            The value of the integral and the error
        """
        if not self.fitted:
            raise RuntimeError(
                'The histogram distribution must be fitted before it can be integrated. Call self.fit(data) to fit the histogram distribution.'
            )

        # Find the bins that contain the integration bounds
        idx_a = np.digitize(a, self._bins)
        idx_b = np.digitize(b, self._bins)

        # Get the bin_width
        bin_width = np.diff(self._bins)[0]

        # Compute the integral
        integral = np.sum(self._hist[idx_a:idx_b]**moment * bin_width)

        # Compute the error
        error = np.sqrt(np.sum(self._hist[idx_a:idx_b]**moment * bin_width)**2)

        return integral, error

    def grad(self, x: float) -> float:
        """
        Compute the derivative of the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to compute the derivative

        Returns
        -------
        float
            The value of the derivative at the given point
        """
        if not self.fitted:
            raise RuntimeError(
                'The histogram distribution must be fitted before it can be differentiated. Call self.fit(data) to fit the histogram distribution.'
            )

        if (x < self._bins[0]) or (x >= self._bins[-1]):
            return 0

        # Find the bin that contains the point
        idx = np.digitize(x, self._bins) - 1

        # Get the bin_width
        bin_width = np.diff(self._bins)[0]

        # Compute the derivative
        def yy(idx: int) -> float:
            if idx > len(self._hist) - 1:
                return 0
            else:
                return self._hist[idx]

        dy = (yy(idx + 1) - yy(idx - 1)) / (2*bin_width)

        return dy


class InterpolateDistribution(BaseDistribution):
    """Use a histogram distribution with interpolated values to estimate a probability density function."""

    def __init__(self, nbins: int = 100):
        """
        Parameters
        ----------
        nbins : int, optional
            The number of bins to use in the histogram (default is 100)
        """
        super().__init__()

        # Set the number of bins
        self.nbins = nbins

    def fit(
        self,
        data: Iterable[float],
        n: int = 1,
        s: float = 0.01,
    ) -> 'InterpolateDistribution':
        """
        Parameters
        ----------
        data : array_like
            The data to fit
        n : int, optional
            The degree of the interpolating polynomial (default is 1)
        s : float, optional
            The smoothing factor (default is 0.01)

        Returns
        -------
        HistogramDistribution
            The fitted histogram distribution
        """
        # Fit the histogram distribution
        self._hist, self._bins = np.histogram(data, bins=self.nbins, density=True)

        # Interpolate
        dx = np.diff(self._bins)[0]
        X = self._bins[:-1] + dx/2
        Y = self._hist

        self._spl = splrep(X, Y, k=n, s=s, per=False)

        # Set the fitted status
        self.fitted = True

        return self

    def __call__(self, x: float) -> float:
        """
        Use the values of nearby bins to interpolate the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point

        Raises
        ------
        RuntimeError
            If the histogram distribution has not been fitted
        """
        if not self.fitted:
            raise RuntimeError(
                'The histogram distribution must be fitted before it can be evaluated. Call self.fit(data) to fit the histogram distribution.'
            )

        if (x < self._bins[0]) or (x >= self._bins[-1]):
            return 0

        return float(splev(x, self._spl))


class EmpiricalDistribution(BaseDistribution):
    """Estimate an empirical distribution using a Gaussian KDE."""

    def __init__(self,
                 bw_method: Optional[Union[str, float, Callable]] = None,
                 weights: Optional[Iterable[float]] = None):
        """
        Parameters
        ----------
        bw_method : str, float, or callable, optional
            The method used to calculate the estimator bandwidth. This can be 'scott', 'silverman', a scalar constant or a callable. If a scalar, this will be used directly as `kde.factor`. If a callable, it should take a `gaussian_kde` instance as only parameter and return a scalar. If None (default), 'scott' is used. See ``scipy.stats.gaussian_kde`` for more details.
        weights : array_like, optional
            The weights for each value in the empirical distribution. This must be the same shape as the input data. See ``scipy.stats.gaussian_kde`` for more details.
        """
        super().__init__()

        # Set the bandwidth method
        self.bw_method = bw_method

        # Set the weights
        self.weights = weights

        # Set the fitted status
        self.fitted = False

    def fit(self, data: Iterable[float]) -> 'EmpiricalDistribution':
        """
        Parameters
        ----------
        data : array_like
            The data to fit

        Returns
        -------
        EmpiricalDistribution
            The fitted empirical distribution
        """
        # Fit the empirical distribution
        self._kernel = gaussian_kde(data,
                                    bw_method=self.bw_method,
                                    weights=self.weights)

        # Set the fitted status
        self.fitted = True

        return self

    def __call__(self, x: float) -> float:
        """
        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point

        Raises
        ------
        RuntimeError
            If the empirical distribution has not been fitted
        """
        if not self.fitted:
            raise RuntimeError(
                'The empirical distribution must be fitted before it can be evaluated. Call self.fit(data) to fit the empirical distribution.'
            )

        return self._kernel(x)

    @property
    def max(self):
        return self._kernel.dataset.max()

    @property
    def min(self):
        return self._kernel.dataset.min()


class WignerSemicircle(BaseDistribution):
    """A Wigner semicircle probability density function."""

    def __init__(self, sigma: float = 1.0):
        """
        The Wigner semicircle (WS) distribution is a probability density function that describes the asymptotic distribution of the eigenvalues of a large symmetric random matrix with Gaussian entries. Given a matrix :math:`X` of size :math:`n \\times n`, where :math:`n \\to \\infty`, the WS distribution is defined as:

        .. math::

            \\mu(x) = \\frac{1}{2 \\pi \\sigma^2} \\sqrt{4 \\sigma^2 - x^2} \\mathbb{1}_{[-2 \\sigma, 2 \\sigma]}(x)

        where :math:`\\mathbb{1}_{[-2 \\sigma, 2 \\sigma]}(x)` is the indicator function of the interval :math:`[-2 \\sigma, 2 \\sigma]`.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian distribution (default is 1.0)
        """
        super().__init__()

        # Set the standard deviation
        self.sigma = sigma

    def __call__(self, x: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """

        # Check the input
        if x <= -2.0 * self.sigma or x >= 2.0 * self.sigma:
            return 0.0

        # Evaluate the probability density function
        func = np.sqrt(4.0 * self.sigma**2 - x**2)
        return func / (2.0 * np.pi * self.sigma**2)  # normalization

    @property
    def max(self) -> float:
        return 2 * self.sigma

    @property
    def min(self) -> float:
        return -2 * self.sigma


class InverseWignerSemicircle(WignerSemicircle):
    """An inverse Wigner semicircle probability density function."""

    def __call__(self, q: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        q : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """
        return super().__call__(1.0 / q) / q**2

    @property
    def max(self) -> float:
        return np.inf

    @property
    def min(self) -> float:
        return -np.inf


class MarchenkoPastur(BaseDistribution):
    """A Marchenko-Pastur probability density function."""

    def __init__(self, L: float, sigma: float = 1.0):
        """
        The Marchenko-Pastur (MP) distribution is a probability density function that describes the asymptotic distribution of the singular values of a large rectangular random matrix with Gaussian entries (i.e. the eigenvalues of the associated covariance matrix). Given a matrix :math:`X` of size :math:`n \\times p`, where :math:`n \\to \\infty` and :math:`p \\to \\infty`, such that :math:`\\lambda = \\frac{p}{n} \\in (0, 1)`, the MP distribution is defined as:

        .. math::

            \\mu(x) = \\frac{1}{2 \\pi \\sigma^2} \\frac{\\sqrt{(\\lambda_+ - x)(x - \\lambda_-)}}{\\lambda x} \\mathbb{1}_{[x_-, x_+]}(x)

        where

        .. math::

            \\lambda_+ = \\sigma^2 (1 + \\sqrt{\\lambda})^2

        and

        .. math::

            \\lambda_- = \\sigma^2 (1 - \\sqrt{\\lambda})^2

        are the upper and lower bounds of the function, respectively, and :math:`\\mathbb{1}_{[x_-, x_+]}(x)` is the indicator function of the interval :math:`[x_-, x_+]`.

        Parameters
        ----------
        L : float
            The ratio :math:`\\lambda = \\frac{p}{n}`, where :math:`p` is the number of variables and :math:`n` is the number of observations of the random matrix.
        sigma : float, optional
            The standard deviation of the random matrix. The default is 1.0.
        """
        super().__init__()

        # Set the ratio
        self.L = L

        # Set the standard deviation
        self.sigma = sigma

        # Define the boundaries of the function
        self.lp = self.sigma**2 * (1 + np.sqrt(self.L))**2
        self.lm = self.sigma**2 * (1 - np.sqrt(self.L))**2

    def __call__(self, x: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """

        if x <= self.lm or x >= self.lp:
            return 0

        numer = np.sqrt((x - self.lm) * (self.lp - x))
        denom = self.L * x
        return numer / denom / (2 * np.pi * self.sigma**2)  # normalization

    @property
    def max(self) -> float:
        return self.lp  # maximum value of the function

    @property
    def min(self) -> float:
        return self.lm  # minimum value of the function


class TranslatedMarchenkoPastur(MarchenkoPastur):
    """A Marchenko-Pastur probability density function with lowest eigenvalue at zero."""

    def __call__(self, x: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """
        return super().__call__(x + self.l0)

    @property
    def max(self) -> float:
        return self.lp - self.lm  # maximum value of the function

    @property
    def min(self) -> float:
        return 0.0  # minimum value of the function

    @property
    def l0(self) -> float:
        return self.lm  # lowest eigenvalue


class InverseMarchenkoPastur(MarchenkoPastur):
    """An inverse Marchenko-Pastur probability density function."""

    def __call__(self, q: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        q : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """
        if q <= self.min or q >= self.max:
            return 0.0
        self.lm = 0.0
        return super().__call__(1.0 / q) / q**2

    @property
    def max(self) -> float:
        return np.inf  # maximum value of the function

    @property
    def min(self) -> float:
        return 1 / self.lp  # minimum value of the function


class TranslatedInverseMarchenkoPastur(InverseMarchenkoPastur):
    """An inverse Marchenko-Pastur probability density function with lowest eigenvalue at zero."""

    def __call__(self, q: float) -> float:
        """
        Evaluate the probability density function at a given point.

        Parameters
        ----------
        q : float
            The point at which to evaluate the probability density function

        Returns
        -------
        float
            The value of the probability density function at the given point
        """
        return super().__call__(q + self.m2)

    @property
    def max(self) -> float:
        return np.inf  # maximum value of the function

    @property
    def min(self) -> float:
        return 0.0  # minimum value of the function

    @property
    def m2(self) -> float:
        return 1 / self.lp  # inverse of the largest eigenvalue
