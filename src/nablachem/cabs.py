# inspired by Psi4's MP2F12::form_cabs_singles implementation to allow for python-only CABS without psi4
# https://github.com/psi4/psi4/blob/e59d050cab7a56306f93c23ee62e482f8b469c0d/psi4/src/psi4/f12/mp2.cc#L218

import re
import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian
from pathlib import Path


def resolve_basis(name):
    """Return a PySCF-compatible basis: string name or {elem: shells} dict from a .gbs file."""
    if isinstance(name, dict):
        return name
    if name.endswith(".gbs"):
        return _parse_gbs_per_element(name)
    return name


def _parse_gbs_per_element(filename):
    """Parse a multi-element Gaussian .gbs file into {elem: shells} dict.

    parse_gaussian.parse() ignores element labels and returns a flat shell list;
    we split the file by element headers first, then parse each section.
    """
    with open(filename) as fh:
        content = fh.read()

    # Strip 'spherical'/'cartesian' directive lines if present (Psi4 adds them)
    content = re.sub(r"(?im)^\s*(spherical|cartesian)\s*$", "", content)

    # Match lines of the form "ELEM 0" (element header in Gaussian format)
    elem_re = re.compile(r"^([A-Z][a-z]?) +0\s*$", re.MULTILINE)
    matches = list(elem_re.finditer(content))

    result = {}
    for i, m in enumerate(matches):
        elem = m.group(1)
        shell_start = m.end()  # after "ELEM 0\n"
        shell_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        shells = parse_gaussian.parse(content[shell_start:shell_end])
        result[elem] = shells

    return result


LINDEP_TOL = 1.0e-8  # threshold for CABS orthogonalization (Psi4 uses 1e-8)


def form_basissets(mol, obs_basis, cabs_basis):
    nobs_ao = mol.nao

    S_obs = mol.intor("int1e_ovlp")
    assert S_obs.shape == (
        nobs_ao,
        nobs_ao,
    ), f"S_obs shape {S_obs.shape} != ({nobs_ao}, {nobs_ao})"

    cabs_dict = resolve_basis(cabs_basis)
    elements = set(a[0] for a in mol._atom)
    missing = elements - set(cabs_dict)
    if missing:
        raise ValueError(f"CABS basis missing entries for elements: {sorted(missing)}")

    def _obs_shells(elem):
        # obs_basis may be a plain basis name (same for every atom) or a per-atom
        # {label: shells} dict, enabling asymmetric basis assignments.
        if isinstance(obs_basis, dict):
            return obs_basis[elem]
        return gto.basis.load(obs_basis, elem)

    combined_basis = {elem: _obs_shells(elem) + cabs_dict[elem] for elem in elements}
    mol_ri = mol.copy()
    mol_ri.basis = combined_basis
    mol_ri.build()

    _S_tmp = gto.intor_cross("int1e_ovlp", mol, mol_ri)
    _obs_idx = np.argmax(np.abs(_S_tmp), axis=1)
    _ao_loc_tmp = mol_ri.ao_loc_nr()
    _obs_sh = sorted(
        set((np.searchsorted(_ao_loc_tmp, _obs_idx, side="right") - 1).tolist())
    )
    _cabs_sh = [s for s in range(mol_ri.nbas) if s not in set(_obs_sh)]
    mol_ri._bas = mol_ri._bas[_obs_sh + _cabs_sh]

    S_mix = gto.intor_cross("int1e_ovlp", mol, mol_ri)
    S_ri = mol_ri.intor("int1e_ovlp")

    e_obs, v_obs = np.linalg.eigh(S_obs)
    keep_obs = e_obs > LINDEP_TOL
    n_orth_obs = int(keep_obs.sum())
    X_obs = v_obs[:, keep_obs] / np.sqrt(e_obs[keep_obs])

    e_ri, v_ri = np.linalg.eigh(S_ri)
    keep_ri = e_ri > LINDEP_TOL
    X_ri = v_ri[:, keep_ri] / np.sqrt(e_ri[keep_ri])

    S_cross = X_ri.T @ S_mix.T @ X_obs

    U, sigma, Vt = np.linalg.svd(S_cross, full_matrices=True)

    cabs_in_orth = U[:, n_orth_obs:]

    C_cabs_ao = X_ri @ cabs_in_orth
    return mol_ri, C_cabs_ao


def form_fock(mf, mol_ri, C_cabs_ao, density_fit=False, dZ=None):
    mol = mf.mol
    C_obs = mf.mo_coeff
    nobs = C_obs.shape[1]
    nocc = mol.nelectron // 2
    nri_ao = mol_ri.nao
    ncabs = C_cabs_ao.shape[1]
    nri = nobs + ncabs
    F_ao = mf.get_fock()

    T_oc = gto.intor_cross("int1e_kin", mol, mol_ri)
    # 0.5 corrects int1e_nuc double-counting nuclei across the combined mol/mol_ri
    # environment (see tests/checks); the geometries are identical.
    V_oc = gto.intor_cross("int1e_nuc", mol, mol_ri) * 0.5
    H_oc = T_oc + V_oc

    T_cc = mol_ri.intor("int1e_kin")
    V_cc = mol_ri.intor("int1e_nuc")
    H_cc = T_cc + V_cc

    if dZ is not None:
        # Alchemical shift of the external potential in the CABS blocks. int1e_rinv
        # is a single-center 1/r operator (no double-counting), so it enters at
        # full weight, unlike the halved int1e_nuc above.
        dZ = np.asarray(dZ, dtype=float)
        coords = mol.atom_coords()  # bohr
        for i, dZi in enumerate(dZ):
            with mol.with_rinv_origin(coords[i]):
                H_oc -= dZi * gto.intor_cross("int1e_rinv", mol, mol_ri)
            with mol_ri.with_rinv_origin(coords[i]):
                H_cc -= dZi * mol_ri.intor("int1e_rinv")

    F_obs_obs = C_obs.T @ F_ao @ C_obs
    F_obs_cabs = C_obs.T @ H_oc @ C_cabs_ao
    F_cabs_cabs = C_cabs_ao.T @ H_cc @ C_cabs_ao

    C_occ = C_obs[:, :nocc]
    D_ao = 2.0 * C_occ @ C_occ.T

    S_mix_ovlp = gto.intor_cross("int1e_ovlp", mol, mol_ri)
    obs_idx = np.argmax(np.abs(S_mix_ovlp), axis=1)

    D_ri = np.zeros((nri_ao, nri_ao))
    D_ri[np.ix_(obs_idx, obs_idx)] = D_ao

    calc_JK = scf.RHF(mol_ri)
    if density_fit:
        calc_JK = calc_JK.density_fit(auxbasis="cc-pvtz-ri")
    J_ri, K_ri = calc_JK.get_jk(mol_ri, D_ri, hermi=1)
    JK_ri = J_ri - 0.5 * K_ri

    JK_ao_oc = JK_ri[obs_idx, :]
    JK_ao_cc = JK_ri

    F_obs_cabs += C_obs.T @ JK_ao_oc @ C_cabs_ao
    F_cabs_cabs += C_cabs_ao.T @ JK_ao_cc @ C_cabs_ao

    f = np.zeros((nri, nri))
    f[:nobs, :nobs] = F_obs_obs
    f[:nobs, nobs:] = F_obs_cabs
    f[nobs:, :nobs] = F_obs_cabs.T
    f[nobs:, nobs:] = F_cabs_cabs

    return f, nocc


def form_cabs_singles(f, nocc):
    f_ij = f[:nocc, :nocc].copy()
    e_ij, C_ij = np.linalg.eigh(f_ij)
    f_AB = f[nocc:, nocc:].copy()
    e_AB, C_AB = np.linalg.eigh(f_AB)
    f_view = f[:nocc, nocc:]
    f_iA = C_ij.T @ f_view @ C_AB
    denom = e_ij[:, None] - e_AB[None, :]
    return 2.0 * np.sum(f_iA**2 / denom)


def CABS_singles_RHF(atomspec, obs_basis, cabs_basis, density_fit=False, dZ=None):
    """CABS singles correction and HF energy for a closed-shell RHF reference.

    Args:
        atomspec: PySCF atom specification (geometry in Angstrom).
        obs_basis: Orbital basis, either a basis name or a per-atom
            ``{label: shells}`` dict for asymmetric assignments.
        cabs_basis: CABS basis (``.gbs`` path or ``{elem: shells}`` dict).
        density_fit: Use density fitting for the J/K build.
        dZ: Optional per-atom nuclear-charge displacement vector. Each nucleus
            ``i`` is alchemically shifted to charge ``Z_i + dZ[i]`` by adding the
            fractional external potential to the core Hamiltonian while keeping
            the basis fixed, consistently across the OBS SCF (``E_HF``) and the
            CABS blocks of the Fock matrix (``E_singles``).
    """
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = obs_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0

    if dZ is not None:
        dZ = np.asarray(dZ, dtype=float)
        if dZ.shape != (mol.natm,):
            raise ValueError(
                f"dZ must have one entry per atom ({mol.natm}), got shape {dZ.shape}"
            )
        coords = mol.atom_coords()  # bohr

        # electronic: extend the external potential by the fractional charge
        h1 = mf.get_hcore()
        s = np.zeros_like(h1)
        for i, dZi in enumerate(dZ):
            with mol.with_rinv_origin(coords[i]):
                s -= dZi * mol.intor("int1e_rinv")
        mf.get_hcore = lambda *args, **kwargs: h1 + s

        # nuclear: difference in nucleus-nucleus repulsion relative to the
        # unperturbed charges (constant shift to the total energy)
        nn = 0.0
        for i in range(mol.natm):
            Z_i = mol.atom_charge(i) + dZ[i]
            for j in range(i + 1, mol.natm):
                Z_j = mol.atom_charge(j) + dZ[j]
                rij = np.linalg.norm(coords[i] - coords[j])
                missing = Z_i * Z_j - mol.atom_charge(i) * mol.atom_charge(j)
                nn += missing / rij
        _kernel = mf.kernel
        mf.kernel = lambda *args, **kwargs: _kernel(*args, **kwargs) + nn

    E_hf = mf.kernel()

    mol_ri, C_cabs_ao = form_basissets(mol, obs_basis, cabs_basis)
    f, nocc = form_fock(mf, mol_ri, C_cabs_ao, density_fit, dZ=dZ)
    E_singles = form_cabs_singles(f, nocc)

    return E_hf, E_singles


def time_rel_to_dz(atomspec):
    CABS_GBS = str(
        Path(__file__).parent.parent.parent / "tests" / "cabs" / "pcseg-cabs.gbs"
    )
    # CABS_GBS = "/Users/guido/Downloads/cabsmodmanual.gbs"
    OBS_BASIS = "pcseg-0"

    import time

    start = time.time()
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = OBS_BASIS
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    mol_ri, C_cabs_ao = form_basissets(mol, OBS_BASIS, CABS_GBS)
    f, nocc = form_fock(mf, mol_ri, C_cabs_ao)
    form_cabs_singles(f, nocc)
    stop = time.time()
    elapsed = stop - start

    # dz comparison
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = "cc-pvdz"
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    start = time.time()
    mf.kernel()  # includes SCF iterations, J/K build, etc.
    stop = time.time()
    elapsed_dz = stop - start

    return elapsed / elapsed_dz
