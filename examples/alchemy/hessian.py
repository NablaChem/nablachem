# %%
from nablachem.alchemy import Anygrad
import numpy as np
import pyscf.gto
import pyscf.scf

atomspec = "C 0 0 0; O 0 0 1.1"
basis = "6-31G"

mf = pyscf.scf.RHF(pyscf.gto.M(atom=atomspec, basis=basis, symmetry=False, verbose=0))
mf.kernel()

ag = Anygrad(mf, Anygrad.Property.ENERGY)
grad_nuc = ag.get("Z")
hess_nuc = ag.get("Z", "Z")
grad_spatial = ag.get("R")
hess_spatial = ag.get("R", "R")
hess_mixed = ag.get("Z", "R")

gradient = np.concatenate([grad_nuc, grad_spatial])
hessian = np.block([[hess_nuc, hess_mixed], [hess_mixed.T, hess_spatial]])

print(np.round(gradient, 2))
print(np.round(hessian, 2))
