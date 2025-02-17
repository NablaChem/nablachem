# code originally from Giorgio Domenichini, https://github.com/giorgiodomen/Supplementary_code_for_Quantum_Alchemy
# Reference: Alchemical predictions of relaxed geometries throughout chemical space by Giorgio Domenichini and O.Anatole von Lilienfeld.
# revised and updated Guido von Rudorff

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.scf import cphf
from pyscf.hessian.rhf import gen_vind

NUC_FRAC_CHARGE = gto.mole.NUC_FRAC_CHARGE
NUC_MOD_OF = gto.mole.NUC_MOD_OF
PTR_FRAC_CHARGE = gto.mole.PTR_FRAC_CHARGE
PTR_ZETA = gto.mole.PTR_ZETA


def g1(mol0, dP, P, DZ, g0):  # dP/dz*dH/dx, P* d2H/dzdx
    natm = mol0.natm
    nao = mol0.nao
    denv = mol0._env.copy()
    datm = mol0._atm.copy()
    datm[:, NUC_MOD_OF] = NUC_FRAC_CHARGE
    for i in range(natm):
        datm[i, PTR_FRAC_CHARGE] = datm[i, PTR_ZETA]
        denv[datm[i, PTR_FRAC_CHARGE]] = DZ[i]
    dH1 = -gto.moleintor.getints(
        "int1e_ipnuc_sph", datm, mol0._bas, denv, None, 3, 0, "s1"
    )
    dH_dxdz = np.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        with mol0.with_rinv_at_nucleus(atm_id):
            vrinv = -mol0.intor("int1e_iprinv", comp=3)
        shl0, shl1, p0, p1 = mol0.aoslice_by_atom()[atm_id]
        vrinv *= DZ[atm_id]
        vrinv[:, p0:p1] += dH1[:, p0:p1]
        vrinv += vrinv.transpose(0, 2, 1)
        dH_dxdz[atm_id] = vrinv
    ga_1 = np.zeros((natm, 3))
    for i in range(natm):
        ga_1[i] += np.einsum("xij,ij->x", g0.hcore_generator()(i), dP)
        ga_1[i] += np.einsum("xij,ij->x", dH_dxdz[i], P)
    return ga_1


def g2(mol0, dP, P, DZ, g0):
    natm = mol0.natm
    nao = mol0.nao
    aoslices = mol0.aoslice_by_atom()
    ga_2 = np.zeros((natm, 3))
    vhf = g0.get_veff(mol0, P)
    vhf_1 = g0.get_veff(mol0, P + dP)
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        ga_2[ia] = np.einsum("xij,ij->x", vhf[:, p0:p1], dP[p0:p1]) * 2
        ga_2[ia] += (
            np.einsum("xij,ij->x", vhf_1[:, p0:p1] - vhf[:, p0:p1], P[p0:p1]) * 2
        )
    return ga_2


def g3(mol0, dP, P, g0, e, e1, C, dC):  # -dW/dZ *dS/dx
    s1 = g0.get_ovlp(mol0)
    g3 = np.zeros((mol0.natm, 3))
    nocc = mol0.nelec[0]
    dW = (
        np.einsum("i,ji,ki->jk", 2 * e[:nocc], dC[:, :nocc], C[:, :nocc])
        + np.einsum("i,ji,ki->jk", 2 * e[:nocc], C[:, :nocc], dC[:, :nocc])
        + 2 * C[:, :nocc] @ e1 @ C.T[:nocc, :]
    )
    for i in range(mol0.natm):
        p0, p1 = mol0.aoslice_by_atom()[i, 2:]
        g3[i] -= np.einsum("xij,ij->x", s1[:, p0:p1], dW[p0:p1]) * 2
    return g3


# with CPHF done
def aaff_resolv(mf, DZ, U, dP, e1):
    mol0 = mf.mol
    g0 = mf.Gradients()
    P = mf.make_rdm1()
    C = mf.mo_coeff
    e = mf.mo_energy
    dC = C @ U
    return (
        g1(mol0, dP, P, DZ, g0)
        + g2(mol0, dP, P, DZ, g0)
        + g3(mol0, dP, P, g0, e, e1, C, dC)
    )


def alc_deriv_grad_nuc(
    mol, dL
):  # to get the derivative with respect to alch. perturbation
    gn = np.zeros((mol.natm, 3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1 - r2)
                gn[j] += (q1 * dL[j] + q2 * dL[i]) * (r1 - r2) / r**3
    return gn


def alc_differential_grad_nuc(
    mol, dL
):  # to get the exact diffeential after alch. perturbation
    gn = np.zeros((mol.natm, 3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j) + dL[j]
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i) + dL[i]
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1 - r2)
                gn[j] += (
                    (q1 * q2 - mol.atom_charge(i) * mol.atom_charge(j))
                    * (r1 - r2)
                    / r**3
                )
    return gn


def g3_old(mol0, dP, P, DZ, g0, vresp, F):  # -dW/dZ *dS/dx
    s1 = g0.get_ovlp(mol0)
    dF = vresp(dP) + DeltaV(mol0, DZ)
    S = mol0.intor_symmetric("int1e_ovlp")
    S_1 = np.linalg.inv(S)
    g3 = np.zeros((mol0.natm, 3))
    for i in range(mol0.natm):
        p0, p1 = mol0.aoslice_by_atom()[i, 2:]
        g3[i] -= (
            np.einsum("xij,ij->x", s1[:, p0:p1], (S_1 @ ((F @ dP) + (dF @ P)))[p0:p1])
            * 2
        )
    return g3


max_cycle_cphf = 40  # default PYSCF params
conv_tol_cphf = 1e-9  # default PYSCF params


def alchemy_cphf_deriv(mf, int_r, with_cphf=True):
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, ~occidx]
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    h1 = lib.einsum(
        "pq,pi,qj->ij", int_r, mo_coeff.conj(), orbo
    )  # going to molecular orbitals
    h1 = h1.reshape((1, h1.shape[0], h1.shape[1]))
    s1 = np.zeros_like(h1)
    vind = gen_vind(mf, mo_coeff, mo_occ)
    mo1, e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1, max_cycle_cphf, conv_tol_cphf)
    return mo1[0], e1[0]


def first_deriv_nuc_nuc(mol, dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    dnn = 0
    for j in range(len(dL[0])):
        r2 = mol.atom_coord(dL[0][j])
        for i in range(mol.natm):
            if i != dL[0][j]:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1 - r2)
                dnn += (q1 * dL[1][j]) / r
    return dnn


def second_deriv_nuc_nuc(mol, dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    dnn = 0
    for j in range(len(dL[0])):
        r2 = mol.atom_coord(dL[0][j])
        for i in range(len(dL[0])):
            if dL[0][i] > dL[0][j]:
                r1 = mol.atom_coord(dL[0][i])
                r = np.linalg.norm(r1 - r2)
                dnn += (dL[1][i] * dL[1][j]) / r
    return 2 * dnn


def first_deriv_elec(mf, int_r):
    P = mf.make_rdm1()
    return np.einsum("ij,ji", P, int_r)


def second_deriv_elec(mf, int_r, mo1):
    orbo = mf.mo_coeff[:, : mo1.shape[1]]
    h1 = lib.einsum("pq,pi,qj->ij", int_r, mf.mo_coeff.conj(), orbo)
    e2 = np.einsum("pi,pi", h1, mo1)
    e2 *= 4
    return e2


def third_deriv_elec(mf, int_r, mo1, e1):  # only for one site (d^3 E /dZ^3)
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    mo1 = lib.einsum("qi,pq->pi", mo1, mo_coeff)
    dm1 = lib.einsum("pi,qi->pq", mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(1, 0)
    vresp = mf.gen_response(hermi=1)  # (J-K/2)(dm)
    h1ao = int_r + vresp(dm1)  # Fock matrix
    e3 = lib.einsum("pq,pi,qi", h1ao, mo1, mo1) * 2  # *2 for double occupancy
    e3 -= lib.einsum("pq,pi,qj,ij", mf.get_ovlp(), mo1, mo1, e1) * 2
    e3 *= 6
    return e3


def alch_deriv(mf, dL=[]):
    """alch_deriv(mf,dL=[]) returns U,dP for a dl=.001 times the charges
    dL can be the whole list of nuclear charges placed on atom, with length equals to mol.natm (eg.[0,1,0,0,-1,...,0])
    or alternatively a list with two sublist of equal length in the form [[atm_idxs],[atm_charges]]
    """
    mol = mf.mol
    dL = parse_charge(dL)
    int_r = DeltaV(mol, dL)
    mo1, e1 = alchemy_cphf_deriv(mf, int_r)
    der1 = first_deriv_elec(mf, int_r) + first_deriv_nuc_nuc(mol, dL)
    der2 = second_deriv_elec(mf, int_r, mo1) + second_deriv_nuc_nuc(mol, dL)
    der3 = third_deriv_elec(mf, int_r, mo1, e1)
    return (der1, der2, der3)


def make_dP(mf, mo1):
    mol = mf.mol
    nao = mol.nao
    nocc = mf.mol.nelec[0]
    C = mf.mo_coeff
    dP = np.zeros_like(C)
    dP[:, :] = 2 * np.einsum("ij,jk,lk->il", C, mo1, C[:, :nocc])
    return dP + dP.T


def make_U(mo1):
    U = np.zeros((mo1.shape[0], mo1.shape[0]))
    U[:, : mo1.shape[1]] = mo1
    U = U - U.T
    return U


def alch_hessian(mf, int_r, mo1):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    h1 = lib.einsum("xpq,pi,qj->xij", int_r, mo_coeff.conj(), orbo)
    e2 = np.einsum("xpi,ypi->xy", h1, mo1)
    e2 = (e2 + e2.T) * 2
    return e2


def cubic_alch_hessian(mf, int_r, mo1, e1):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    mo1 = lib.einsum("xqi,pq->xpi", mo1, mo_coeff)  # dC=UC
    dm1 = lib.einsum("xpi,qi->xpq", mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(0, 2, 1)  # dP= dCOC^T+COdC'T
    vresp = mf.gen_response(hermi=1)
    h1ao = int_r + vresp(dm1)  # dF=dV+G(dP)
    # *2 for double occupancy
    e3 = lib.einsum("xpq,ypi,zqi->xyz", h1ao, mo1, mo1) * 2  # trace( dC^T dF dC)
    e3 -= (
        lib.einsum("pq,xpi,yqj,zij->xyz", mf.get_ovlp(), mo1, mo1, e1) * 2
    )  # - dC^T S dC de
    e3 = (
        e3
        + e3.transpose(1, 2, 0)
        + e3.transpose(2, 0, 1)
        + e3.transpose(0, 2, 1)
        + e3.transpose(1, 0, 2)
        + e3.transpose(2, 1, 0)
    )
    return e3


charge2symbol = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
}


def alias_param(param_name: str, param_alias: str):
    """
    Decorator for aliasing a param in a function
    Args:
        param_name: name of param in function to alias
        param_alias: alias that can be used for this param
    Returns:
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            alias_param_value = kwargs.get(param_alias)
            if param_alias in kwargs.keys():
                kwargs[param_name] = alias_param_value
                del kwargs[param_alias]
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def printxyz(coords, al, fn):
    atomnumber = {1: "H", 5: "B", 6: "C", 7: "N"}
    assert len(al) == len(coords)
    with open(fn, "w") as xyzf:
        xyzf.write(str(len(al)) + " \n")
        xyzf.write("molecule \n")
        for i in range(len(coords)):
            xyzf.write(atomnumber[al[i]] + "    " + str(coords[i])[1:-1] + "\n")
    return


def parse_charge(dL):
    """There are two options:
    1) call FcM(**kwargs,fcs=[c1,c2,--cn]) with a list of length equal to the number of atoms
    2) FcM(**kwargs,fcs=[[aidx1,aidx2,..,aidxn],[c1,c2,..cn]]) with a list of two sublist for atoms' indexes and fract charges
    """
    a = [[], []]
    parsed = False
    if len(dL) == 2:  # necessario, ma non sufficiente per caso 2
        try:
            len(dL[0]) == len(dL[1])
            if isinstance(dL[0][0], int) or isinstance(dL[0][0], float):
                parsed = True
        except:
            pass
    if not parsed and (
        isinstance(dL[0], int) or isinstance(dL[0], float)
    ):  # move to case2
        for i in range(len(dL)):
            if dL[i] != 0:
                a[0].append(i)
                a[1].append(dL[i])
        dL = a
        parsed = True
    if not parsed:
        print("Failed to parse charges")
        raise
    return dL


def DeltaV(mol, dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    mol.set_rinv_orig_(mol.atom_coords()[dL[0][0]])
    dV = mol.intor("int1e_rinv") * dL[1][0]
    for i in range(1, len(dL[0])):
        mol.set_rinv_orig_(mol.atom_coords()[dL[0][i]])
        dV += mol.intor("int1e_rinv") * dL[1][i]
    return -dV


def with_rinv_at_nucleus(self, atm_id):
    rinv = self.atom_coord(atm_id)
    self._env[gto.mole.AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
    return self.with_rinv_origin(rinv)


class FracMole(gto.Mole):
    def with_rinv_at_nucleus(self, atm_id):
        rinv = self.atom_coord(atm_id)
        self._env[gto.mole.AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
        return self.with_rinv_origin(rinv)


def FcM(fcs=[], **kwargs):
    mol = FracMole()
    mol.build(**kwargs)
    init_charges = mol.atom_charges()
    if fcs:
        fcs = parse_charge(fcs)
        init_charges = mol.atom_charges()
        for j in range(len(fcs[0])):
            mol._env[mol._atm[fcs[0][j], PTR_FRAC_CHARGE]] = (
                init_charges[fcs[0][j]] + fcs[1][j]
            )
            mol._atm[fcs[0][j], NUC_MOD_OF] = NUC_FRAC_CHARGE
        mol.charge = sum(fcs[1]) + mol.charge
    return mol


def FcM_like(in_mol, fcs=[]):
    mol = in_mol.copy()
    mol.with_rinv_at_nucleus = with_rinv_at_nucleus.__get__(mol)
    mol.symmetry = None  # symmetry usually breaks after perturbation
    init_charges = mol.atom_charges()
    if fcs:
        fcs = parse_charge(fcs)
        init_charges = mol.atom_charges()
        for j in range(len(fcs[0])):
            mol._env[mol._atm[fcs[0][j], PTR_FRAC_CHARGE]] = (
                init_charges[fcs[0][j]] + fcs[1][j]
            )
            mol._atm[fcs[0][j], NUC_MOD_OF] = NUC_FRAC_CHARGE
        mol.charge = in_mol.charge + sum(fcs[1])
    return mol


class APDFT_perturbator(lib.StreamObject):
    def __init__(self, mf, symmetry=None, sites=None):
        self.mf = mf
        self.mol = mf.mol
        self.symm = symmetry
        self.sites = []
        for site in sites:
            self.sites.append(site)
        self.DeltaV = DeltaV
        self.alchemy_cphf_deriv = alchemy_cphf_deriv
        self.make_dP = make_dP
        self.make_U = make_U
        self.dVs = {}
        self.mo1s = {}
        self.e1s = {}
        self.dPs = {}
        self.afs = {}
        self.perturb()
        self.cubic_hessian = None
        self.hessian = None
        self.gradient = None
        self.xcf = None
        try:
            self.xcf = mf.xc
        except:
            pass

    def U(self, atm_idx):
        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()
        return make_U(self.mo1s[atm_idx])

    def dP(self, atm_idx):
        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()
        return make_dP(self.mf, self.mo1s[atm_idx])

    def dP_atom(self, atm_idx):
        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()
        return make_dP(self.mf, self.mo1s[atm_idx])

    def dP_pred(self, pvec):
        pvec = np.asarray(pvec)
        return self.mf.make_rdm1() + np.array(
            [self.dP_atom(i) for i in self.sites]
        ).transpose(1, 2, 0).dot(pvec)

    def perturb(self):
        for site in self.sites:
            if site in self.mo1s:
                pass
            elif self.symm and site in self.symm.eqs:
                ref_idx = self.symm.eqs[site]["ref"]
                if ref_idx in self.mo1s:
                    self.dVs[site] = DeltaV(self.mol, [[site], [1]])
                    self.mo1s[site], self.e1s[site] = self.symm.rotate_mo1e1(
                        self.mo1s[ref_idx],
                        self.e1s[ref_idx],
                        site,
                        ref_idx,
                        self.mf.mo_coeff,
                        self.mf.get_ovlp(),
                    )
                else:
                    continue
            else:
                self.dVs[site] = DeltaV(self.mol, [[site], [1]])
                self.mo1s[site], self.e1s[site] = alchemy_cphf_deriv(
                    self.mf, self.dVs[site]
                )

    def mo1(self, atm_idx):
        if atm_idx not in self.mo1s:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.mo1s[atm_idx]

    def dV(self, atm_idx):
        if atm_idx not in self.dVs:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.dVs[atm_idx]

    def e1(self, atm_idx):
        if atm_idx not in self.e1s:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.e1s[atm_idx]

    def af(self, atm_idx):
        if atm_idx in self.afs:
            return self.afs[atm_idx]
        elif self.symm and atm_idx in self.symm.eqs:
            ref_idx = self.symm.eqs[atm_idx]["ref"]
            afr = self.af(ref_idx)
            self.afs[atm_idx] = self.symm.symm_gradient(afr, atm_idx, ref_idx)
        else:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
            DZ = [0 for x in range(self.mol.natm)]
            DZ[atm_idx] = 1
            af = aaff_resolv(
                self.mf, DZ, U=self.U(atm_idx), dP=self.dP(atm_idx), e1=self.e1(atm_idx)
            )
            af += alc_deriv_grad_nuc(self.mol, DZ)
            self.afs[atm_idx] = af
        return self.afs[atm_idx]

    def first_deriv(self, atm_idx):
        return first_deriv_elec(self.mf, self.dV(atm_idx)) + first_deriv_nuc_nuc(
            self.mol, [[atm_idx], [1]]
        )

    def second_deriv(self, idx_1, idx_2):
        return second_deriv_elec(
            self.mf, self.dV(idx_1), self.mo1(idx_2)
        ) + second_deriv_nuc_nuc(self.mol, [[idx_1, idx_2], [1, 1]])

    def third_deriv(self, pvec):
        pvec = np.asarray(pvec)
        return np.einsum("ijk,i,j,k", self.cubic_hessian, pvec, pvec, pvec)

    def build_gradient(self):
        idxs = self.sites
        self.gradient = np.asarray([self.first_deriv(x) for x in idxs])
        return self.gradient

    def build_hessian(self):
        mo1s = []
        dVs = []
        for id in self.sites:
            mo1s.append(self.mo1(id))
            dVs.append(self.dV(id))
        mo1s = np.asarray(mo1s)
        dVs = np.asarray(dVs)
        self.hessian = alch_hessian(self.mf, dVs, mo1s) + self.hessian_nuc_nuc(
            *self.sites
        )
        return self.hessian

    def hessian_nuc_nuc(self, *args):
        idxs = []
        for arg in args:
            if isinstance(arg, int):
                idxs.append(arg)
        hessian = np.zeros((len(idxs), len(idxs)))
        for i in range(len(idxs)):
            for j in range(i, len(idxs)):
                hessian[i, j] = (
                    second_deriv_nuc_nuc(self.mol, [[idxs[i], idxs[j]], [1, 1]]) / 2
                )
        hessian += hessian.T
        return hessian

    def build_cubic_hessian(self):
        idxs = self.sites
        mo1s = np.asarray([self.mo1(x) for x in idxs])
        dVs = np.asarray([self.dV(x) for x in idxs])
        e1s = np.asarray([self.e1(x) for x in idxs])
        self.cubic_hessian = cubic_alch_hessian(self.mf, dVs, mo1s, e1s)
        return self.cubic_hessian

    def build_all(self):
        self.build_gradient()
        self.build_hessian()
        self.build_cubic_hessian()


def parse_to_array(natm, dL):
    arr = np.zeros(natm)
    for i in range(len(dL[0])):
        arr[dL[0][i]] = dL[1][i]
    return arr
