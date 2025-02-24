import scipy.linalg

scipy.linalg._solve = scipy.linalg.solve
scipy.linalg.solve = lambda *args, **kwargs: scipy.linalg._solve(*args)

import pyscf.gto
import pyscf.scf

atomspec = "C 0 0 0; O 0 0 1.1"
basis = "unc-ccpvdz"

mf = pyscf.scf.RHF(pyscf.gto.M(atom=atomspec, basis=basis, symmetry=False))
mf.kernel()


from AP_class import APDFT_perturbator as AP


ap_nn = AP(mf, sites=[0, 1])
# %%
print(ap_nn.af(0)[:, -1], ap_nn.af(1)[:, -1])

print(
    """(array([[ 2.29761672e-13,  5.53667464e-13, -2.61185786e-01],
        [-2.29761672e-13, -5.53667464e-13,  2.61185786e-01]]),
 array([[ 2.07093170e-13,  2.24561870e-13, -3.85009559e-02],
        [-2.07093170e-13, -2.24561870e-13,  3.85009559e-02]]))"""
)
