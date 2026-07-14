"""ProtoRep -- global representation from geometry + a minimal-basis Hamiltonian.

Two size-extensive, element-pair-resolved blocks summed over all atom pairs,
then standardized:

  geom : a fine Gaussian radial distribution function (bond-length resolved),
         one channel per element pair.
  ham  : Frobenius norms of the minimal-basis one-electron Hamiltonian
         (``H_core = T + V_nuc``) and overlap (``S``) blocks between each atom
         pair -- genuine quantum-chemical bonding strengths that vary
         continuously with nuclear charge -- plus a per-element on-site
         ``||H_AA||``.  Built from one-electron integrals only (no SCF).

Requires PySCF for the Hamiltonian block.
"""

import numpy as np

from .features import BaseRepresenter


def _hamiltonian(mol, basis):
    """Return (H_core, S, atom AO slices) in the given minimal basis."""
    from pyscf import gto

    atoms = [(int(z), tuple(p))
             for z, p in zip(mol.get_atomic_numbers(), mol.get_positions())]
    m = gto.M(atom=atoms, basis=basis, verbose=0, unit="Angstrom")
    H = m.intor("int1e_kin") + m.intor("int1e_nuc")
    S = m.intor("int1e_ovlp")
    return H, S, m.aoslice_by_atom()[:, 2:4]


class ProtoRep(BaseRepresenter):
    basis = "sto-3g"
    rcut = 6.0          # interaction cutoff (Angstrom)
    ngauss = 24         # number of radial Gaussians
    rmin, rmax = 0.7, 3.2

    def _prepare(self, molecules):
        self._elements = sorted({int(z) for m in molecules
                                 for z in m.get_atomic_numbers()})
        self._eidx = {e: i for i, e in enumerate(self._elements)}
        self._pair = {}
        for i, a in enumerate(self._elements):
            for b in self._elements[i:]:
                self._pair[(a, b)] = len(self._pair)
        self._mu = np.linspace(self.rmin, self.rmax, self.ngauss)
        dr = (self.rmax - self.rmin) / (self.ngauss - 1)
        self._eta = 1.0 / (2.0 * dr ** 2)
        X = np.stack([self._featurize(m) for m in molecules])
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

    def _featurize(self, mol):
        Z = mol.get_atomic_numbers()
        pos = mol.get_positions()
        npair = len(self._pair)
        geom = np.zeros((npair, self.ngauss))
        hoff = np.zeros((npair, 2))
        H, S, sl = _hamiltonian(mol, self.basis)
        i, j = np.triu_indices(len(Z), k=1)
        d = np.linalg.norm(pos[i] - pos[j], axis=1) if len(i) else np.empty(0)
        keep = d <= self.rcut
        i, j, d = i[keep], j[keep], d[keep]
        for k in range(len(d)):
            a, b = sorted((int(Z[i[k]]), int(Z[j[k]])))
            p = self._pair[(a, b)]
            geom[p] += np.exp(-self._eta * (d[k] - self._mu) ** 2)
            ai, bi = slice(*sl[i[k]]), slice(*sl[j[k]])
            hoff[p, 0] += np.linalg.norm(H[ai, bi])
            hoff[p, 1] += np.linalg.norm(S[ai, bi])
        hon = np.zeros(len(self._elements))
        for a in range(len(Z)):
            hon[self._eidx[int(Z[a])]] += np.linalg.norm(H[slice(*sl[a]), slice(*sl[a])])
        return np.concatenate([geom.ravel(), hoff.ravel(), hon])

    def compute(self, molecules):
        return [(self._featurize(m) - self._mean) / self._std for m in molecules]
