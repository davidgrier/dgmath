import unittest
import numpy as np
from scipy.stats import norm
from dgmath.kde import kde, akde


class TestKde(unittest.TestCase):
    def test_1d(self):
        '''Gaussian distributed random points'''
        mu = 5.     # mean
        sigma = 1.  # variance
        npts = 1000
        x = np.random.normal(loc=mu, scale=sigma, size=npts)
        y = np.linspace(0, 10)

        # True distribution
        rho_true = norm(mu, sigma).pdf(y)

        # KDE estimate
        rho_kde = kde(x, y, stats=False)

        tolerance = 3./np.sqrt(npts)
        self.assertTrue(np.allclose(rho_true, rho_kde,
                                    atol=tolerance,
                                    rtol=tolerance))

    def test_2d(self):
        '''Gaussian peak in 2d'''
        mux = 0.
        muy = 0.
        sigmax = 1.0
        sigmay = 0.5
        npts = 1000
        xx = np.random.normal(loc=mux, scale=sigmax, size=npts)
        xy = np.random.normal(loc=muy, scale=sigmay, size=npts)
        data = [xx, xy]
        yx = np.random.normal(loc=mux, scale=sigmax, size=100)
        yy = np.random.normal(loc=muy, scale=sigmay, size=100)
        sample = [yx, yy]

        # True distribution
        rho_true = (norm(mux, sigmax).pdf(yx) *
                    norm(muy, sigmay).pdf(yy))

        # KDE estimate
        rho_kde = kde(data, sample)

        tolerance = 3./np.sqrt(npts)
        self.assertTrue(np.allclose(rho_true, rho_kde,
                                    atol=tolerance,
                                    rtol=tolerance))


class TestAKde(unittest.TestCase):
    def test_1d(self):
        '''Gaussian distributed random points'''
        mu = 5.     # mean
        sigma = 1.  # variance
        npts = 1000
        x = np.random.normal(loc=mu, scale=sigma, size=npts)
        y = np.linspace(0, 10)

        # True distribution
        rho_true = norm(mu, sigma).pdf(y)

        # AKDE estimate
        rho_kde = akde(x, y)

        tolerance = 3./np.sqrt(npts)
        self.assertTrue(np.allclose(rho_true, rho_kde,
                                    atol=tolerance,
                                    rtol=tolerance))

    def test_2d(self):
        '''Gaussian peak in 2d'''
        mux = 0.
        muy = 0.
        sigmax = 1.0
        sigmay = 0.5
        npts = 1000
        xx = np.random.normal(loc=mux, scale=sigmax, size=npts)
        xy = np.random.normal(loc=muy, scale=sigmay, size=npts)
        data = [xx, xy]
        yx = np.random.normal(loc=mux, scale=sigmax, size=100)
        yy = np.random.normal(loc=muy, scale=sigmay, size=100)
        sample = [yx, yy]

        # True distribution
        rho_true = (norm(mux, sigmax).pdf(yx) *
                    norm(muy, sigmay).pdf(yy))

        # KDE estimate
        rho_kde = akde(data, sample)

        tolerance = 3./np.sqrt(npts)
        result = np.allclose(rho_true, rho_kde,
                             atol=tolerance,
                             rtol=tolerance)
        if result is False:
            print(np.max(np.abs(rho_true-rho_kde)))
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
