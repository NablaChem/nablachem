# inspired by Psi4's MP2F12::form_cabs_singles implementation to allow for python-only CABS without psi4
# https://github.com/psi4/psi4/blob/e59d050cab7a56306f93c23ee62e482f8b469c0d/psi4/src/psi4/f12/mp2.cc#L218

import re
import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian


def resolve_basis(name):
    """Return a PySCF-compatible basis: string name or {elem: shells} dict from a .gbs file."""
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


# ---------------------------------------------------------------------------
# System: water monomer
# ---------------------------------------------------------------------------

MOL_XYZ = """
O   0.000000   0.000000   0.117176
H   0.000000   0.757329  -0.468706
H   0.000000  -0.757329  -0.468706
"""
OBS_BASIS = "pcseg0"
CABS_BASIS = "/Users/guido/Downloads/cabstwo.gbs"
LINDEP_TOL = 1.0e-8  # threshold for CABS orthogonalization (Psi4 uses 1e-8)


# ---------------------------------------------------------------------------
# Step 1 — form_basissets
# Build CABS: orthogonalize RI basis against OBS, keep complement
# Mirrors: OrbitalSpace::build_cabs_space / build_ri_space
# ---------------------------------------------------------------------------


def form_basissets(mol, obs_basis, cabs_basis):
    """
    obs_basis  : OBS basis set name (string, e.g. "pcseg0")
    cabs_basis : path to a .gbs file for the CABS basis
    Returns:
        mol_ri    : pyscf Mole for the RI/CABS AO basis
        C_cabs_ao : (nri_ao, ncabs) coefficients mapping RI AOs -> CABS MOs,
                    orthogonal to OBS
        nobs_ao   : number of OBS AO basis functions (= mol.nao)
        ncabs     : number of CABS functions surviving orthogonalization
    """
    nobs_ao = mol.nao

    # OBS overlap — needed to orthonormalise OBS
    S_obs = mol.intor("int1e_ovlp")  # (nobs_ao, nobs_ao)
    assert S_obs.shape == (
        nobs_ao,
        nobs_ao,
    ), f"S_obs shape {S_obs.shape} != ({nobs_ao}, {nobs_ao})"

    # RI basis: OBS + CABS blended (mirrors Psi4 "Blend: PCSEG-0 + CABSTWO")
    # Psi4 builds the RI space as OBS ∪ CABS_FILE (79 AOs for this system),
    # then projects out OBS to get CABS.  Using only CABS_FILE here would give
    # a smaller RI space and wrong CABS.
    cabs_dict = resolve_basis(cabs_basis)
    elements = set(a[0] for a in mol._atom)
    combined_basis = {
        elem: gto.basis.load(obs_basis, elem) + cabs_dict[elem] for elem in elements
    }
    mol_ri = mol.copy()
    mol_ri.basis = combined_basis
    mol_ri.build()
    nri_ao = mol_ri.nao

    # Reorder mol_ri shells so OBS shells come first (0..n_obs_sh-1), then CABS.
    # This enables a tight shls_slice restriction in form_fock: the J/K density-
    # matrix contraction runs over only OBS shells, not all RI shells.
    _S_tmp = gto.intor_cross("int1e_ovlp", mol, mol_ri)
    _obs_idx = np.argmax(np.abs(_S_tmp), axis=1)
    _ao_loc_tmp = mol_ri.ao_loc_nr()
    _obs_sh = sorted(set((np.searchsorted(_ao_loc_tmp, _obs_idx, side="right") - 1).tolist()))
    _cabs_sh = [s for s in range(mol_ri.nbas) if s not in set(_obs_sh)]
    mol_ri._bas = mol_ri._bas[_obs_sh + _cabs_sh]

    # Mixed overlap S[mu_obs, mu_ri] and RI overlap
    S_mix = gto.intor_cross("int1e_ovlp", mol, mol_ri)  # (nobs_ao, nri_ao)
    S_ri = mol_ri.intor("int1e_ovlp")  # (nri_ao, nri_ao)
    assert S_mix.shape == (
        nobs_ao,
        nri_ao,
    ), f"S_mix shape {S_mix.shape} != ({nobs_ao}, {nri_ao})"
    assert S_ri.shape == (
        nri_ao,
        nri_ao,
    ), f"S_ri shape {S_ri.shape} != ({nri_ao}, {nri_ao})"

    # Löwdin orthonormalization of OBS and RI independently
    e_obs, v_obs = np.linalg.eigh(S_obs)
    keep_obs = e_obs > LINDEP_TOL
    n_orth_obs = int(keep_obs.sum())
    X_obs = v_obs[:, keep_obs] / np.sqrt(e_obs[keep_obs])  # (nobs_ao, n_orth_obs)
    assert X_obs.shape == (
        nobs_ao,
        n_orth_obs,
    ), f"X_obs shape {X_obs.shape} != ({nobs_ao}, {n_orth_obs})"

    e_ri, v_ri = np.linalg.eigh(S_ri)
    keep_ri = e_ri > LINDEP_TOL
    n_orth_ri = int(keep_ri.sum())
    X_ri = v_ri[:, keep_ri] / np.sqrt(e_ri[keep_ri])  # (nri_ao, n_orth_ri)
    assert X_ri.shape == (
        nri_ao,
        n_orth_ri,
    ), f"X_ri shape {X_ri.shape} != ({nri_ao}, {n_orth_ri})"
    assert (
        n_orth_ri >= n_orth_obs
    ), f"RI space ({n_orth_ri}) must be larger than OBS ({n_orth_obs})"

    # Overlap between the two orthonormal spaces: (n_orth_ri, n_orth_obs)
    # S_cross[p, q] = <phi_ri_p | phi_obs_q>  in orthonormal bases
    S_cross = X_ri.T @ S_mix.T @ X_obs  # (n_orth_ri, n_orth_obs)
    assert S_cross.shape == (
        n_orth_ri,
        n_orth_obs,
    ), f"S_cross shape {S_cross.shape} != ({n_orth_ri}, {n_orth_obs})"

    # SVD of S_cross.
    # Left singular vectors (U) span the RI space.
    # Columns of U beyond rank(S_cross) ~ n_orth_obs are the CABS:
    # the part of the RI orthonormal space not covered by OBS.
    U, sigma, Vt = np.linalg.svd(S_cross, full_matrices=True)
    assert U.shape == (
        n_orth_ri,
        n_orth_ri,
    ), f"U shape {U.shape} != ({n_orth_ri}, {n_orth_ri})"
    assert Vt.shape == (
        n_orth_obs,
        n_orth_obs,
    ), f"Vt shape {Vt.shape} != ({n_orth_obs}, {n_orth_obs})"
    assert (
        len(sigma) == n_orth_obs
    ), f"sigma length {len(sigma)} != n_orth_obs {n_orth_obs}"

    # CABS = left null space of S_cross = U[:, n_orth_obs:]
    ncabs = n_orth_ri - n_orth_obs
    cabs_in_orth = U[:, n_orth_obs:]  # (n_orth_ri, ncabs)
    assert cabs_in_orth.shape == (
        n_orth_ri,
        ncabs,
    ), f"cabs_in_orth shape {cabs_in_orth.shape} != ({n_orth_ri}, {ncabs})"
    assert ncabs > 0, "No CABS functions survive orthogonalization"

    # Back-transform to RI AO basis
    C_cabs_ao = X_ri @ cabs_in_orth  # (nri_ao, ncabs)
    assert C_cabs_ao.shape == (
        nri_ao,
        ncabs,
    ), f"C_cabs_ao shape {C_cabs_ao.shape} != ({nri_ao}, {ncabs})"

    # Sanity: CABS must be orthogonal to OBS (overlap with any OBS function ~ 0)
    overlap_cabs_obs = C_cabs_ao.T @ S_mix.T @ np.eye(nobs_ao)  # (ncabs, nobs_ao)
    assert np.allclose(
        overlap_cabs_obs, 0, atol=1e-6
    ), f"CABS not orthogonal to OBS, max |overlap| = {np.abs(overlap_cabs_obs).max():.2e}"

    return mol_ri, C_cabs_ao, nobs_ao, ncabs


# ---------------------------------------------------------------------------
# Step 2 — form_fock
# Build the full Fock matrix in the RI MO basis (OBS MOs + CABS)
# Mirrors: MP2F12::form_fock  (form_oeints + J/K contraction)
#
# Layout (same as Psi4 nri_ × nri_ tensor):
#   rows/cols 0 .. nobs-1      : OBS MOs  (occupied + virtual)
#   rows/cols nobs .. nri-1    : CABS functions
# ---------------------------------------------------------------------------


def form_fock(mf, mol_ri, C_cabs_ao):
    """
    mf        : converged pyscf RHF object on the OBS mol
    mol_ri    : pyscf Mole for the RI/CABS AO basis (same geometry)
    C_cabs_ao : (nri_ao, ncabs) CABS coefficients in RI AO basis

    Returns:
        f     : (nri, nri) Fock matrix in (OBS MOs | CABS) basis
        nocc  : number of occupied MOs
        nobs  : number of OBS MOs  (= mol.nao for a full basis)
        ncabs : number of CABS functions
        nri   : nobs + ncabs
    """
    mol = mf.mol
    C_obs = mf.mo_coeff  # (nobs_ao, nobs)
    nobs_ao = mol.nao
    nobs = C_obs.shape[1]  # number of OBS MOs (= nobs_ao)
    nocc = mol.nelectron // 2
    nri_ao = mol_ri.nao
    ncabs = C_cabs_ao.shape[1]
    nri = nobs + ncabs

    assert C_obs.shape == (
        nobs_ao,
        nobs,
    ), f"C_obs shape {C_obs.shape} != ({nobs_ao}, {nobs})"
    assert C_cabs_ao.shape == (
        nri_ao,
        ncabs,
    ), f"C_cabs_ao shape {C_cabs_ao.shape} != ({nri_ao}, {ncabs})"
    assert nocc < nobs, f"nocc {nocc} must be < nobs {nobs}"

    # AO Fock matrix from converged HF (includes J and K)
    F_ao = mf.get_fock()  # (nobs_ao, nobs_ao)
    assert F_ao.shape == (
        nobs_ao,
        nobs_ao,
    ), f"F_ao shape {F_ao.shape} != ({nobs_ao}, {nobs_ao})"

    # --- One-electron integrals (mirrors form_oeints) ---
    # OBS–CABS block: T and V in mixed basis  (nobs_ao, nri_ao)
    T_oc = gto.intor_cross("int1e_kin", mol, mol_ri)
    # NOTE: intor_cross("int1e_nuc", mol, mol_ri) internally concatenates _atm
    # from both mols (mol has O,H,H and mol_ri = mol.copy() also has O,H,H).
    # The nuclear potential is therefore summed over 6 atoms = 2× the correct
    # value.  Since mol_ri is an exact copy of mol (same atoms, same positions),
    # dividing by 2 recovers the correct <mu_obs|V_nuc|nu_cabs>.
    V_oc = gto.intor_cross("int1e_nuc", mol, mol_ri) * 0.5
    H_oc = T_oc + V_oc  # (nobs_ao, nri_ao)
    assert H_oc.shape == (
        nobs_ao,
        nri_ao,
    ), f"H_oc shape {H_oc.shape} != ({nobs_ao}, {nri_ao})"

    # CABS–CABS block: T and V in RI AO basis (nri_ao, nri_ao)
    T_cc = mol_ri.intor("int1e_kin")
    V_cc = mol_ri.intor("int1e_nuc")
    H_cc = T_cc + V_cc  # (nri_ao, nri_ao)
    assert H_cc.shape == (
        nri_ao,
        nri_ao,
    ), f"H_cc shape {H_cc.shape} != ({nri_ao}, {nri_ao})"

    # Transform one-electron blocks to MO basis
    F_obs_obs = C_obs.T @ F_ao @ C_obs  # (nobs, nobs)  — full F including J, K
    F_obs_cabs = C_obs.T @ H_oc @ C_cabs_ao  # (nobs, ncabs) — H only, J/K added below
    F_cabs_cabs = C_cabs_ao.T @ H_cc @ C_cabs_ao  # (ncabs, ncabs) — H only
    assert F_obs_obs.shape == (nobs, nobs), f"F_obs_obs shape {F_obs_obs.shape}"
    assert F_obs_cabs.shape == (nobs, ncabs), f"F_obs_cabs shape {F_obs_cabs.shape}"
    assert F_cabs_cabs.shape == (ncabs, ncabs), f"F_cabs_cabs shape {F_cabs_cabs.shape}"

    # --- Two-electron (J and K) contributions to CABS blocks ---
    # Density matrix from occupied MOs: D = 2 C_occ C_occ^T  (AO, OBS)
    C_occ = C_obs[:, :nocc]  # (nobs_ao, nocc)
    D_ao = 2.0 * C_occ @ C_occ.T  # (nobs_ao, nobs_ao)
    assert D_ao.shape == (
        nobs_ao,
        nobs_ao,
    ), f"D_ao shape {D_ao.shape} != ({nobs_ao}, {nobs_ao})"

    # Compute J, K on mol_ri with OBS-only density matrix.
    # form_basissets reorders mol_ri shells so OBS shells are first (0..n_obs_sh-1),
    # making D_ri's nonzero block contiguous at the top-left.  This helps PySCF's
    # Cauchy-Schwarz screener skip CABS-CABS shell pairs efficiently in one pass.
    # Cross-overlap locates each OBS AO in the reordered mol_ri index space.
    S_mix_ovlp = gto.intor_cross("int1e_ovlp", mol, mol_ri)  # (nobs_ao, nri_ao)
    obs_idx = np.argmax(np.abs(S_mix_ovlp), axis=1)  # (nobs_ao,)

    D_ri = np.zeros((nri_ao, nri_ao))
    D_ri[np.ix_(obs_idx, obs_idx)] = D_ao

    J_ri, K_ri = scf.RHF(mol_ri).get_jk(mol_ri, D_ri, hermi=1)

    # F = H + J - 0.5*K  (factor of 2 already absorbed in D_ri)
    JK_ri = J_ri - 0.5 * K_ri  # (nri_ao, nri_ao)

    # obs_idx rows give the OBS AOs in mol's ordering; all nri_ao cols = full RI
    JK_ao_oc = JK_ri[obs_idx, :]  # (nobs_ao, nri_ao)
    JK_ao_cc = JK_ri  # (nri_ao, nri_ao)
    assert JK_ao_oc.shape == (
        nobs_ao,
        nri_ao,
    ), f"JK_ao_oc shape {JK_ao_oc.shape} != ({nobs_ao}, {nri_ao})"
    assert JK_ao_cc.shape == (
        nri_ao,
        nri_ao,
    ), f"JK_ao_cc shape {JK_ao_cc.shape} != ({nri_ao}, {nri_ao})"

    # Add 2e contributions to the cross/CABS blocks
    F_obs_cabs += C_obs.T @ JK_ao_oc @ C_cabs_ao
    F_cabs_cabs += C_cabs_ao.T @ JK_ao_cc @ C_cabs_ao
    assert F_obs_cabs.shape == (
        nobs,
        ncabs,
    ), f"F_obs_cabs shape after JK: {F_obs_cabs.shape}"
    assert F_cabs_cabs.shape == (
        ncabs,
        ncabs,
    ), f"F_cabs_cabs shape after JK: {F_cabs_cabs.shape}"

    # Assemble the full nri × nri Fock matrix (Psi4 layout)
    f = np.zeros((nri, nri))
    f[:nobs, :nobs] = F_obs_obs
    f[:nobs, nobs:] = F_obs_cabs
    f[nobs:, :nobs] = F_obs_cabs.T
    f[nobs:, nobs:] = F_cabs_cabs
    assert f.shape == (nri, nri), f"f shape {f.shape} != ({nri}, {nri})"
    # assert np.allclose(f, f.T, atol=1e-10), \
    #    f"Fock matrix not symmetric, max |f - f.T| = {np.abs(f - f.T).max():.2e}"

    return f, nocc, nobs, ncabs, nri


# ---------------------------------------------------------------------------
# Step 3 — form_cabs_singles
# Direct translation of MP2F12::form_cabs_singles (mp2.cc:218-255)
# ---------------------------------------------------------------------------


def form_cabs_singles(f, nocc, nobs, ncabs, nri):
    """
    f     : (nri, nri) full Fock matrix in (OBS MOs | CABS) basis
    nocc  : number of occupied orbitals
    nobs  : number of OBS MOs
    ncabs : number of CABS functions
    nri   : nobs + ncabs

    Returns E_singles : CABS singles correction energy (negative)
    """
    all_vir = nri - nocc  # nvir + ncabs

    assert f.shape == (nri, nri), f"f shape {f.shape} != ({nri}, {nri})"
    assert nri == nobs + ncabs, f"nri {nri} != nobs+ncabs {nobs}+{ncabs}"
    assert 0 < nocc < nobs, f"nocc {nocc} out of range [1, nobs={nobs})"
    assert all_vir > 0, f"all_vir = {all_vir} must be > 0"

    # Diagonalize f_ij (occupied-occupied block) — mirrors syev(&C_ij, &e_ij)
    f_ij = f[:nocc, :nocc].copy()
    assert f_ij.shape == (nocc, nocc), f"f_ij shape {f_ij.shape} != ({nocc}, {nocc})"

    e_ij, C_ij = np.linalg.eigh(f_ij)  # (nocc,), (nocc, nocc)
    assert e_ij.shape == (nocc,), f"e_ij shape {e_ij.shape}"
    assert C_ij.shape == (nocc, nocc), f"C_ij shape {C_ij.shape}"

    # Diagonalize f_AB (vir+CABS block) — mirrors syev(&C_AB, &e_AB)
    f_AB = f[nocc:, nocc:].copy()
    assert f_AB.shape == (
        all_vir,
        all_vir,
    ), f"f_AB shape {f_AB.shape} != ({all_vir}, {all_vir})"

    e_AB, C_AB = np.linalg.eigh(f_AB)  # (all_vir,), (all_vir, all_vir)
    assert e_AB.shape == (all_vir,), f"e_AB shape {e_AB.shape}"
    assert C_AB.shape == (all_vir, all_vir), f"C_AB shape {C_AB.shape}"

    # Transform off-diagonal block to eigenbasis — mirrors gemm calls in Psi4
    f_view = f[:nocc, nocc:]  # (nocc, all_vir)
    assert f_view.shape == (
        nocc,
        all_vir,
    ), f"f_view shape {f_view.shape} != ({nocc}, {all_vir})"

    f_iA = C_ij.T @ f_view @ C_AB  # (nocc, all_vir)
    assert f_iA.shape == (
        nocc,
        all_vir,
    ), f"f_iA shape {f_iA.shape} != ({nocc}, {all_vir})"

    # Energy denominators: e_i - e_A.
    # For OBS virtuals this is negative (occupied below virtual), but CABS-derived
    # eigenstates can have eigenvalues anywhere in the spectrum (including below
    # occupied levels for tight/high-l functions), so we do NOT assert sign here.
    # Psi4 sums all terms as-is (mp2.cc:250).
    denom = e_ij[:, None] - e_AB[None, :]  # (nocc, all_vir)
    assert denom.shape == (
        nocc,
        all_vir,
    ), f"denom shape {denom.shape} != ({nocc}, {all_vir})"
    assert not np.any(
        denom == 0
    ), "Zero denominator encountered — degenerate occupied/virtual eigenvalue"

    # Energy sum — mirrors the omp loop in Psi4
    E_singles = 2.0 * np.sum(f_iA**2 / denom)

    return E_singles


def CABS_singles_RHF(atomspec, obs_basis: str, cabs_basis: str):
    """
    Compute the CABS singles correction for a closed-shell (RHF) system.

    Parameters
    ----------
    atomspec  : atom specification accepted by pyscf Mole.atom (string, list of
                tuples, etc.).  Coordinates are assumed to be in Angstrom.
    obs_basis : OBS basis set name as accepted by pyscf (e.g. ``"pcseg-0"``).
    cabs_basis: path to a Gaussian .gbs file for the CABS basis.

    Returns
    -------
    E_hf      : RHF total energy (Hartree)
    E_singles : CABS singles correction (Hartree, typically negative)
    """
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = obs_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    E_hf = mf.e_tot

    mol_ri, C_cabs_ao, nobs_ao, ncabs = form_basissets(mol, obs_basis, cabs_basis)
    f, nocc, nobs, ncabs, nri = form_fock(mf, mol_ri, C_cabs_ao)
    E_singles = form_cabs_singles(f, nocc, nobs, ncabs, nri)

    return E_hf, E_singles
