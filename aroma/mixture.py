"""
This file was taken from the nipy Python library and reduced to only contain
code necessary to define and use the GGM class. Minor changes have been made
to the documentation, in order to follow project convention.

License
-------
Copyright (c) 2006-2018, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NIPY Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Notes
-----
Taken from the nipy version at commit c7ae00435ef1134cee28eb0d31d2f1a253328f6b.
"""
import numpy as np
import scipy.special as sp


def _dichopsi_log(u, v, y, eps=0.00001):
    """ Implements the dichotomic part of the solution of psi(c)-log(c)=y
    """
    if u > v:
        u, v = v, u
    t = (u + v) / 2
    if np.absolute(u - v) < eps:
        return t
    else:
        if sp.psi(t) - np.log(t) > y:
            return _dichopsi_log(u, t, y, eps)
        else:
            return _dichopsi_log(t, v, y, eps)


def _psi_solve(y, eps=0.00001):
    """ Solve psi(c)-log(c)=y by dichotomy
    """
    if y > 0:
        print("y", y)
        raise ValueError("y>0, the problem cannot be solved")
    u = 1.
    if y > sp.psi(u) - np.log(u):
        while sp.psi(u) - np.log(u) < y:
            u *= 2
        u /= 2
    else:
        while sp.psi(u) - np.log(u) > y:
            u /= 2
    return _dichopsi_log(u, 2 * u, y, eps)


def _compute_c(x, z, eps=0.00001):
    """
    this function returns the mle of the shape parameter if a 1D gamma
    density
    """
    eps = 1.e-7
    y = np.dot(z, np.log(x)) / np.sum(z) - np.log(np.dot(z, x) / np.sum(z))
    if y > - eps:
        c = 10
    else:
        c = _psi_solve(y, eps=0.00001)
    return c


def _gaus_dens(mean, var, x):
    """ evaluate the gaussian density (mean,var) at points x
    """
    Q = - (x - mean) ** 2 / (2 * var)
    return 1. / np.sqrt(2 * np.pi * var) * np.exp(Q)


def _gam_dens(shape, scale, x):
    """evaluate the gamma density (shape,scale) at points x

    Notes
    -----
    Returns 0 on negative subspace
    """
    ng = np.zeros(np.size(x))
    cst = - shape * np.log(scale) - sp.gammaln(shape)
    i = np.ravel(np.nonzero(x > 0))
    if np.size(i) > 0:
        lz = cst + (shape - 1) * np.log(x[i]) - x[i] / scale
        ng[i] = np.exp(lz)
    return ng


def _gam_param(x, z):
    """ Compute the parameters of a gamma density from data weighted points

    Parameters
    ----------
    x: array of shape(nbitem) the learning points
    z: array of shape(nbitem), their membership within the class

    Notes
    -----
    if no point is positive then the couple (1, 1) is returned
    """
    eps = 1.e-5
    i = np.ravel(np.nonzero(x > 0))
    szi = np.sum(z[i])
    if szi > 0:
        shape = _compute_c(x[i], z[i], eps)
        scale = np.dot(x[i], z[i]) / (szi * shape)
    else:
        shape = 1
        scale = 1
    return shape, scale


class GGM(object):
    """
    This is the basic one dimensional Gaussian-Gamma Mixture estimation class
    Note that it can work with positive or negative values,
    as long as there is at least one positive value.
    NB : The gamma distribution is defined only on positive values.

    5 scalar members
    - mean: gaussian mean
    - var: gaussian variance (non-negative)
    - shape: gamma shape (non-negative)
    - scale: gamma scale (non-negative)
    - mixt: mixture parameter (non-negative, weight of the gamma)
    """

    def __init__(self, shape=1, scale=1, mean=0, var=1, mixt=0.5):
        self.shape = shape
        self.scale = scale
        self.mean = mean
        self.var = var
        self.mixt = mixt

    def parameters(self):
        """ print the paramteres of self
        """
        print("Gaussian: mean: ", self.mean, "variance: ", self.var)
        print("Gamma: shape: ", self.shape, "scale: ", self.scale)
        print("Mixture gamma: ", self.mixt, "Gaussian: ", 1 - self.mixt)

    def Mstep(self, x, z):
        """
        Mstep of the model: maximum likelihood
        estimation of the parameters of the model

        Parameters
        ----------
        x  : array of shape (nbitems,)
            input data
        z array of shape(nbitrems, 2)
            the membership matrix
        """
        # z[0,:] is the likelihood to be generated by the gamma
        # z[1,:] is the likelihood to be generated by the gaussian

        tiny = 1.e-15
        sz = np.maximum(tiny, np.sum(z, 0))

        self.shape, self.scale = _gam_param(x, z[:, 0])
        self.mean = np.dot(x, z[:, 1]) / sz[1]
        self.var = np.dot((x - self.mean) ** 2, z[:, 1]) / sz[1]
        self.mixt = sz[0] / np.size(x)

    def Estep(self, x):
        """
        E step of the estimation:
        Estimation of ata membsership

        Parameters
        ----------
        x: array of shape (nbitems,)
            input data

        Returns
        -------
        z: array of shape (nbitems, 2)
            the membership matrix
        """
        eps = 1.e-15
        z = np.zeros((np.size(x), 2), 'd')
        z[:, 0] = _gam_dens(self.shape, self.scale, x)
        z[:, 1] = _gaus_dens(self.mean, self.var, x)
        z = z * np.array([self.mixt, 1. - self.mixt])
        sz = np.maximum(np.sum(z, 1), eps)
        L = np.sum(np.log(sz)) / np.size(x)
        z = (z.T / sz).T
        return z, L

    def estimate(self, x, niter=10, delta=0.0001, verbose=False):
        """ Complete EM estimation procedure

        Parameters
        ----------
        x : array of shape (nbitems,)
            the data to be processed
        niter : int, optional
            max nb of iterations
        delta : float, optional
            criterion for convergence
        verbose : bool, optional
            If True, print values during iterations

        Returns
        -------
        LL, float
            average final log-likelihood
        """
        if x.max() < 0:
            # all the values are generated by the Gaussian
            self.mean = np.mean(x)
            self.var = np.var(x)
            self.mixt = 0.
            L = 0.5 * (1 + np.log(2 * np.pi * self.var))
            return L

        # proceed with standard estimate
        z, L = self.Estep(x)
        L0 = L - 2 * delta
        for i in range(niter):
            self.Mstep(x, z)
            z, L = self.Estep(x)
            if verbose:
                print(i, L)
            if (L < L0 + delta):
                break
            L0 = L
        return L

    def show(self, x):
        """ Visualization of the mm based on the empirical histogram of x

        Parameters
        ----------
        x : array of shape (nbitems,)
            the data to be processed
        """
        step = 3.5 * np.std(x) / np.exp(np.log(np.size(x)) / 3)
        bins = max(10, int((x.max() - x.min()) / step))
        h, c = np.histogram(x, bins)
        h = h.astype(np.float) / np.size(x)
        p = self.mixt

        dc = c[1] - c[0]
        y = (1 - p) * _gaus_dens(self.mean, self.var, c) * dc
        z = np.zeros(np.size(c))
        z = _gam_dens(self.shape, self.scale, c) * p * dc

        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(0.5 * (c[1:] + c[:-1]), h)
        mp.plot(c, y, 'r')
        mp.plot(c, z, 'g')
        mp.plot(c, z + y, 'k')
        mp.title('Fit of the density with a Gamma-Gaussians mixture')
        mp.legend(('data', 'gaussian acomponent', 'gamma component',
                   'mixture distribution'))

    def posterior(self, x):
        """Posterior probability of observing the data x for each component

        Parameters
        ----------
        x: array of shape (nbitems,)
            the data to be processed

        Returns
        -------
        y, pg : arrays of shape (nbitem)
            the posterior probability
        """
        p = self.mixt
        pg = p * _gam_dens(self.shape, self.scale, x)
        y = (1 - p) * _gaus_dens(self.mean, self.var, x)
        return y / (y + pg), pg / (y + pg)
