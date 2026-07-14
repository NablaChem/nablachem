"""PhysPot -- global representation from element-pair physical pair potentials.

A molecule is described by element-pair-resolved sums of a few energy-like
radial functions evaluated over every atom pair::

    F[(A,B), k] = sum_{i<j, {Zi,Zj}=={A,B}} f_k(r_ij)

with ``f_k`` drawn from the bonding overlap ``exp(-r)``, electrostatics
``1/r``, dispersion ``1/r**6`` and a handful of Gaussians spanning the bonding
region.  Each sum is size-extensive, mirroring the additive structure of the
interaction energy; features are standardized so a single-length-scale kernel
weighs them evenly.  Pure geometry + nuclear charges; no external dependencies.
"""

import numpy as np

from .features import BaseRepresenter

# centers of the bonding-region Gaussian channels (Angstrom)
_MU = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.8])


class PhysPot(BaseRepresenter):
    rcut = 6.0   # interaction cutoff (Angstrom)
    eta = 8.0    # Gaussian width parameter

    def _prepare(self, molecules):
        elements = sorted({int(z) for m in molecules for z in m.get_atomic_numbers()})
        self._pair = {}
        for i, a in enumerate(elements):
            for b in elements[i:]:
                self._pair[(a, b)] = len(self._pair)
        X = np.stack([self._featurize(m) for m in molecules])
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

    def _funcs(self, d):
        cols = [np.exp(-d), 1.0 / d, 1.0 / d ** 6]
        cols += [np.exp(-self.eta * (d - mu) ** 2) for mu in _MU]
        return np.stack(cols, axis=1)

    def _featurize(self, mol):
        Z = mol.get_atomic_numbers()
        pos = mol.get_positions()
        F = np.zeros((len(self._pair), 3 + len(_MU)))
        i, j = np.triu_indices(len(Z), k=1)
        d = np.linalg.norm(pos[i] - pos[j], axis=1) if len(i) else np.empty(0)
        keep = d <= self.rcut
        i, j, d = i[keep], j[keep], d[keep]
        if len(d):
            f = self._funcs(d)
            for k in range(len(d)):
                a, b = sorted((int(Z[i[k]]), int(Z[j[k]])))
                F[self._pair[(a, b)]] += f[k]
        return F.ravel()

    def compute(self, molecules):
        return [(self._featurize(m) - self._mean) / self._std for m in molecules]
