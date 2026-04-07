# inspired by Psi4's MP2F12::form_cabs_singles implementation to allow for python-only CABS without psi4
# https://github.com/psi4/psi4/blob/e59d050cab7a56306f93c23ee62e482f8b469c0d/psi4/src/psi4/f12/mp2.cc#L218

import re
import multiprocessing as mp
import numpy as np
from scipy.optimize import basinhopping, differential_evolution
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian
from pathlib import Path

# Module-level globals set in each spawned worker via _pool_init
_pool_hf_cache: list = []
_pool_obs_basis: str = ""
_pool_cbs_estimates: list = []


class _MockSCF:
    """Lightweight stand-in for pyscf RHF that carries precomputed arrays."""

    def __init__(self, mol, F_ao, mo_coeff):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self._F_ao = F_ao

    def get_fock(self):
        return self._F_ao


def _pool_init(hf_cache, obs_basis, cbs_estimates):
    """Initializer for spawned worker processes: populate module-level globals."""
    global _pool_hf_cache, _pool_obs_basis, _pool_cbs_estimates
    _pool_hf_cache = hf_cache
    _pool_obs_basis = obs_basis
    _pool_cbs_estimates = cbs_estimates


def _hf_fill_worker(args):
    """Run HF for one molecule; returns picklable (E_hf, mo_coeff, F_ao)."""
    atomspec, obs_basis = args
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = obs_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot, mf.mo_coeff.copy(), mf.get_fock()


def _worker_compute_one(args):
    """Compute E_HF + E_singles - E_CBS for one molecule in a spawned worker."""
    i, basis = args
    atomspec, F_ao, mo_coeff, E_hf = _pool_hf_cache[i]
    mol = gto.Mole()
    mol.atom = atomspec
    mol.basis = _pool_obs_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mock_mf = _MockSCF(mol, F_ao, mo_coeff)
    mol_ri, C_cabs_ao = form_basissets(mol, _pool_obs_basis, basis)
    f, nocc = form_fock(mock_mf, mol_ri, C_cabs_ao)
    return i, E_hf + form_cabs_singles(f, nocc) - _pool_cbs_estimates[i]


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
    combined_basis = {
        elem: gto.basis.load(obs_basis, elem) + cabs_dict[elem] for elem in elements
    }
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


def form_fock(mf, mol_ri, C_cabs_ao, density_fit=False):
    mol = mf.mol
    C_obs = mf.mo_coeff
    nobs = C_obs.shape[1]
    nocc = mol.nelectron // 2
    nri_ao = mol_ri.nao
    ncabs = C_cabs_ao.shape[1]
    nri = nobs + ncabs
    F_ao = mf.get_fock()

    T_oc = gto.intor_cross("int1e_kin", mol, mol_ri)
    V_oc = gto.intor_cross("int1e_nuc", mol, mol_ri) * 0.5
    H_oc = T_oc + V_oc

    T_cc = mol_ri.intor("int1e_kin")
    V_cc = mol_ri.intor("int1e_nuc")
    H_cc = T_cc + V_cc

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


def CABS_singles_RHF(atomspec, obs_basis: str, cabs_basis: str, density_fit=False):
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

    mol_ri, C_cabs_ao = form_basissets(mol, obs_basis, cabs_basis)
    f, nocc = form_fock(mf, mol_ri, C_cabs_ao, density_fit)
    E_singles = form_cabs_singles(f, nocc)

    return E_hf, E_singles


def CABS_opt(
    atomspecs: list[str],
    cbs_estimates: list[float],
    obs_basis: str,
    cabs_basis: str,
    output_path: str,
):
    L_NAMES = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}

    def pyscf_to_primitives(basis_dict):
        result = {}
        for elem, shells in basis_dict.items():
            result[elem] = []
            for shell in shells:
                l = shell[0]
                exp = float(shell[1][0])  # first element of first primitive tuple
                result[elem].append((l, exp))
        return result

    def primitives_to_pyscf(prims):
        return {
            elem: [[l, [exp, 1.0]] for l, exp in shells]
            for elem, shells in prims.items()
        }

    def write_gbs(prims, path):
        lines = []
        for elem in sorted(prims):
            lines.append("****")
            lines.append(f"{elem} 0")
            for l, exp in prims[elem]:
                lines.append(f"{L_NAMES[l]}   1   1.00")
                lines.append(f"      {exp:>20.10f}   1.0000000000")
        lines.append("****")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    basis_dict = resolve_basis(cabs_basis)
    primitives = pyscf_to_primitives(basis_dict)

    hf_cache = []
    for i, atomspec in enumerate(atomspecs):
        mol = gto.Mole()
        mol.atom = atomspec
        mol.basis = obs_basis
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        hf_cache.append((mol, mf, mf.e_tot))

    def compute_rmse(prims):
        basis = primitives_to_pyscf(prims)
        errors = []
        for (mol, mf, E_hf), cbs in zip(hf_cache, cbs_estimates):
            mol_ri, C_cabs_ao = form_basissets(mol, obs_basis, basis)
            f, nocc = form_fock(mf, mol_ri, C_cabs_ao)
            E_singles = form_cabs_singles(f, nocc)
            errors.append(E_hf + E_singles - cbs)
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    current_rmse = compute_rmse(primitives)

    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")
        for elem in sorted(primitives):
            for k in range(len(primitives[elem])):
                l, exp = primitives[elem][k]
                print(
                    f"  Scanning {elem} {L_NAMES[l]} exp={exp:.8f}  (RMSE={current_rmse*1000:.6f} mHa)"
                )
                best_rmse = current_rmse
                best_exp = exp

                def try_direction(sign, start=1):
                    nonlocal best_rmse, best_exp
                    for p in (sign * i for i in range(start, 10_000)):
                        trial_exp = exp * (1.1**p)
                        trial_prims = {e: list(primitives[e]) for e in primitives}
                        trial_prims[elem][k] = (l, trial_exp)
                        rmse = compute_rmse(trial_prims)
                        marker = " <--" if rmse < best_rmse else ""
                        print(
                            f"    1.1^{p:+d} exp={trial_exp:.8f}: RMSE={rmse*1000:.6f} mHa{marker}"
                        )
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_exp = trial_exp
                        else:
                            break  # no improvement — stop this direction

                # Try increasing first; if the very first step fails, try decreasing
                trial_exp = exp * 1.1
                trial_prims = {e: list(primitives[e]) for e in primitives}
                trial_prims[elem][k] = (l, trial_exp)
                rmse = compute_rmse(trial_prims)
                marker = " <--" if rmse < best_rmse else ""
                print(
                    f"    1.1^+1 exp={trial_exp:.8f}: RMSE={rmse*1000:.6f} mHa{marker}"
                )
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_exp = trial_exp
                    try_direction(+1, start=2)
                else:
                    try_direction(-1)

                if best_exp != exp:
                    primitives[elem][k] = (l, best_exp)
                    current_rmse = best_rmse
                    print(f"    -> updated: {exp:.8f} -> {best_exp:.8f}")
        print(f"  End of cycle {cycle + 1}: RMSE = {current_rmse * 1000:.6f} mHa")

    write_gbs(primitives, output_path)
    print(f"\nOptimized basis written to: {output_path}")


def _count_elements(atomspec: str) -> dict:
    """Return {element: count} from a PySCF atom string."""
    counts = {}
    for line in atomspec.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        elem = parts[0].capitalize()
        counts[elem] = counts.get(elem, 0) + 1
    return counts


def CABS_opt_detrended(
    atomspecs: list[str],
    cbs_estimates: list[float],
    obs_basis: str,
    cabs_basis: str,
    output_path: str,
):
    """Like CABS_opt but minimises RMSE of (E_HF + E_singles - E_CBS) after
    linear detrending with elemental composition.  Per-element coefficients are
    re-fitted at every candidate basis evaluation."""
    L_NAMES = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}

    def pyscf_to_primitives(basis_dict):
        result = {}
        for elem, shells in basis_dict.items():
            result[elem] = []
            for shell in shells:
                l = shell[0]
                exp = float(shell[1][0])
                result[elem].append((l, exp))
        return result

    def primitives_to_pyscf(prims):
        return {
            elem: [[l, [exp, 1.0]] for l, exp in shells]
            for elem, shells in prims.items()
        }

    def write_gbs(prims, path):
        lines = []
        for elem in sorted(prims):
            lines.append("****")
            lines.append(f"{elem} 0")
            for l, exp in prims[elem]:
                lines.append(f"{L_NAMES[l]}   1   1.00")
                lines.append(f"      {exp:>20.10f}   1.0000000000")
        lines.append("****")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    basis_dict = resolve_basis(cabs_basis)
    primitives = pyscf_to_primitives(basis_dict)

    # Build stoichiometry matrix once (row per molecule, col per element)
    all_elements = sorted({e for spec in atomspecs for e in _count_elements(spec)})
    X = np.zeros((len(atomspecs), len(all_elements)))
    for i, spec in enumerate(atomspecs):
        counts = _count_elements(spec)
        for j, elem in enumerate(all_elements):
            X[i, j] = counts.get(elem, 0)

    _spawn = mp.get_context("spawn")

    # Parallel HF fill: workers return only picklable arrays (no mol/mf objects)
    with _spawn.Pool() as hf_pool:
        hf_results = hf_pool.map(
            _hf_fill_worker,
            [(spec, obs_basis) for spec in atomspecs],
        )

    # picklable_hf_cache[i] = (atomspec, F_ao, mo_coeff, E_hf)
    picklable_hf_cache = [
        (spec, F_ao, mo_coeff, E_hf)
        for spec, (E_hf, mo_coeff, F_ao) in zip(atomspecs, hf_results)
    ]

    # Optimisation pool: each worker receives the full cache once via initializer
    pool = _spawn.Pool(
        initializer=_pool_init,
        initargs=(picklable_hf_cache, obs_basis, list(cbs_estimates)),
    )

    # Map each element to the indices of molecules that contain it.
    # When scanning an exponent for element X, only those molecules need
    # recomputation; all others retain their cached error contribution.
    elem_mask = {}
    for i, atomspec in enumerate(atomspecs):
        for e in _count_elements(atomspec):
            elem_mask.setdefault(e, []).append(i)

    def _compute_errors_for(prims, indices, out):
        """Recompute out[i] for each i in indices using prims; leave others unchanged."""
        basis = primitives_to_pyscf(prims)
        for i, val in pool.map(_worker_compute_one, [(i, basis) for i in indices]):
            out[i] = val

    def _rmse_from_errors(errors):
        alpha, _, _, _ = np.linalg.lstsq(X, errors, rcond=None)
        residuals = errors - X @ alpha
        return float(np.sqrt(np.mean(residuals**2)))

    error_cache = np.zeros(len(atomspecs))
    _compute_errors_for(primitives, range(len(atomspecs)), error_cache)
    current_rmse = _rmse_from_errors(error_cache)

    COARSE_GRID = [0.5, 1, 5, 10, 50, 100]
    STEP_FACTOR = 1.1

    print("\n--- Coarse grid search ---")
    for elem in sorted(primitives):
        for k in range(len(primitives[elem])):
            l, exp = primitives[elem][k]
            best_rmse = current_rmse
            best_exp = exp
            best_errors = error_cache.copy()
            for trial_exp in COARSE_GRID:
                trial_prims = {e: list(primitives[e]) for e in primitives}
                trial_prims[elem][k] = (l, trial_exp)
                trial_errors = error_cache.copy()
                _compute_errors_for(trial_prims, elem_mask.get(elem, []), trial_errors)
                rmse = _rmse_from_errors(trial_errors)
                marker = " <--" if rmse < best_rmse else ""
                print(
                    f"  {elem} {L_NAMES[l]} exp={trial_exp}: RMSE={rmse*1000:.6f} mHa{marker}"
                )
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_exp = trial_exp
                    best_errors = trial_errors
            if best_exp != exp:
                primitives[elem][k] = (l, best_exp)
                current_rmse = best_rmse
                error_cache[:] = best_errors
                print(f"  -> updated: {exp:.8f} -> {best_exp:.8f}")
    print(f"  After coarse search: RMSE = {current_rmse * 1000:.6f} mHa")

    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")
        for elem in sorted(primitives):
            for k in range(len(primitives[elem])):
                l, exp = primitives[elem][k]
                print(
                    f"  Scanning {elem} {L_NAMES[l]} exp={exp:.8f}  (RMSE={current_rmse*1000:.6f} mHa)"
                )
                best_rmse = current_rmse
                best_exp = exp
                best_errors = error_cache.copy()

                def _trial(trial_exp):
                    """Evaluate a trial exponent; return (rmse, errors)."""
                    trial_prims = {e: list(primitives[e]) for e in primitives}
                    trial_prims[elem][k] = (l, trial_exp)
                    trial_errors = error_cache.copy()
                    _compute_errors_for(
                        trial_prims, elem_mask.get(elem, []), trial_errors
                    )
                    return _rmse_from_errors(trial_errors), trial_errors

                def scan_exponent():
                    nonlocal best_rmse, best_exp, best_errors

                    for sign in (+1, -1):
                        for p_abs in range(1, 10_000):
                            p = sign * p_abs
                            trial_exp = exp * (STEP_FACTOR**p)
                            rmse, trial_errors = _trial(trial_exp)
                            marker = " <--" if rmse < best_rmse else ""
                            print(
                                f"    {STEP_FACTOR}^{p:+d} exp={trial_exp:.8f}: RMSE={rmse*1000:.6f} mHa{marker}"
                            )
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_exp = trial_exp
                                best_errors = trial_errors
                            else:
                                break

                scan_exponent()

                if best_exp != exp:
                    primitives[elem][k] = (l, best_exp)
                    current_rmse = best_rmse
                    error_cache[:] = best_errors
                    print(f"    -> updated: {exp:.8f} -> {best_exp:.8f}")
        print(f"  End of cycle {cycle + 1}: RMSE = {current_rmse * 1000:.6f} mHa")

    pool.close()
    pool.join()

    write_gbs(primitives, output_path)
    print(f"\nOptimized basis written to: {output_path}")


def cabs_opt_buildup(
    atomspecs: list[str],
    cbs_estimates: list[float],
    obs_basis: str,
    output_path: str,
    improvement_threshold: float = 0.01,
):
    """Build a CABS basis from scratch by greedy one-function-at-a-time addition.

    Cycles round-robin over elements.  For each element, scans an even-tempered
    log-uniform exponent grid (0.05–500, 10 points) over all relevant angular
    momenta (S only for H; S/P/D/F for all others) and identifies the (l, exp)
    pair that best reduces the stoichiometry-detrended RMSE.  A candidate is
    accepted when it improves RMSE by at least *improvement_threshold*
    (default 5 %) relative to the current value.  After each acceptance the
    existing exponents of that element are relaxed with ±10 % multiplicative
    steps (one pass per exponent, break on first non-improvement, same as
    CABS_opt_detrended).  Molecules whose elements are not yet covered by the
    growing basis contribute E_HF − E_CBS (zero CABS correction) to the error
    array.  The loop stops when a full round over all elements adds nothing.
    """
    L_NAMES = {0: "S", 1: "P", 2: "D", 3: "F"}
    EXP_GRID = np.logspace(np.log10(0.05), np.log10(500), 30)
    STEP_FACTOR = 1.1

    def angular_momenta(elem):
        return [0] if elem == "H" else [0, 1, 2, 3]

    def primitives_to_pyscf(prims):
        return {
            elem: [[l, [exp, 1.0]] for l, exp in shells]
            for elem, shells in prims.items()
        }

    def write_gbs(prims, path):
        lines = []
        for elem in sorted(prims):
            lines.append("****")
            lines.append(f"{elem} 0")
            for l, exp in prims[elem]:
                lines.append(f"{L_NAMES[l]}   1   1.00")
                lines.append(f"      {exp:>20.10f}   1.0000000000")
        lines.append("****")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # --- one-time setup ---
    all_elements = sorted({e for spec in atomspecs for e in _count_elements(spec)})

    X = np.zeros((len(atomspecs), len(all_elements)))
    for i, spec in enumerate(atomspecs):
        counts = _count_elements(spec)
        for j, elem in enumerate(all_elements):
            X[i, j] = counts.get(elem, 0)

    elem_mask = {}
    for i, atomspec in enumerate(atomspecs):
        for e in _count_elements(atomspec):
            elem_mask.setdefault(e, []).append(i)

    _spawn = mp.get_context("spawn")
    with _spawn.Pool() as hf_pool:
        hf_results = hf_pool.map(
            _hf_fill_worker,
            [(spec, obs_basis) for spec in atomspecs],
        )

    picklable_hf_cache = [
        (spec, F_ao, mo_coeff, E_hf)
        for spec, (E_hf, mo_coeff, F_ao) in zip(atomspecs, hf_results)
    ]
    E_hf_list = [entry[3] for entry in picklable_hf_cache]

    pool = _spawn.Pool(
        initializer=_pool_init,
        initargs=(picklable_hf_cache, obs_basis, list(cbs_estimates)),
    )

    def rmse_from_errors(errors):
        alpha, _, _, _ = np.linalg.lstsq(X, errors, rcond=None)
        residuals = errors - X @ alpha
        return float(np.sqrt(np.mean(residuals**2)))

    def complete_prims(prims):
        """Add an empty shell list for every element not yet in *prims*.

        This lets form_basissets see all elements (using only OBS for the
        uncovered ones) so that every molecule can be evaluated regardless of
        how many elements have CABS functions so far.
        """
        full = {elem: [] for elem in all_elements}
        for e, shells in prims.items():
            full[e] = list(shells)
        return full

    def eval_trial(trial_prims, affected_indices, base_errors):
        """Recompute errors[i] for each i in affected_indices; rest unchanged."""
        trial_errors = base_errors.copy()
        if not affected_indices:
            return trial_errors
        trial_pyscf = primitives_to_pyscf(trial_prims)
        for i, val in pool.map(
            _worker_compute_one, [(i, trial_pyscf) for i in affected_indices]
        ):
            trial_errors[i] = val
        return trial_errors

    # --- initial state: empty basis, error = E_HF − E_CBS ---
    basis = {}  # elem -> list of (l, exp)
    error_cache = np.array(
        [e - c for e, c in zip(E_hf_list, cbs_estimates)], dtype=float
    )
    current_rmse = rmse_from_errors(error_cache)
    print(f"Initial RMSE (no CABS): {current_rmse * 1000:.4f} mHa")

    # --- build-up loop ---
    round_num = 0
    while True:
        round_num += 1
        print(f"\n=== Round {round_num} ===")
        any_added = False

        for elem in all_elements:
            print(f"\n  [{elem}] RMSE = {current_rmse * 1000:.4f} mHa")
            best_rmse = current_rmse
            best_candidate = None  # (l, exp)
            best_errors = None

            for l in angular_momenta(elem):
                l_best_rmse = current_rmse
                l_best_exp = None

                for exp in EXP_GRID:
                    trial_prims = complete_prims(basis)
                    trial_prims[elem] = trial_prims[elem] + [(l, exp)]
                    trial_errors = eval_trial(
                        trial_prims, elem_mask.get(elem, []), error_cache
                    )
                    rmse = rmse_from_errors(trial_errors)
                    if rmse < l_best_rmse:
                        l_best_rmse = rmse
                        l_best_exp = exp
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_candidate = (l, exp)
                        best_errors = trial_errors

                if l_best_exp is not None:
                    print(
                        f"    {L_NAMES[l]}: best exp={l_best_exp:.4f},"
                        f" RMSE={l_best_rmse * 1000:.4f} mHa"
                    )
                else:
                    print(f"    {L_NAMES[l]}: no improvement")

            if best_candidate is None:
                print(f"  -> no candidate improves RMSE for {elem}")
                continue

            l, exp = best_candidate
            improvement = (current_rmse - best_rmse) / current_rmse
            print(
                f"  -> best: {L_NAMES[l]} exp={exp:.4f},"
                f" RMSE={best_rmse * 1000:.4f} mHa,"
                f" improvement={improvement * 100:.1f}%"
            )

            if improvement < improvement_threshold:
                print(f"  -> skipped (< {improvement_threshold * 100:.0f}%)")
                continue

            # accept
            basis.setdefault(elem, []).append((l, exp))
            error_cache[:] = best_errors
            current_rmse = best_rmse
            any_added = True
            print(
                f"  -> ACCEPTED. {elem}: "
                f"{[(L_NAMES[ll], f'{e:.4f}') for ll, e in basis[elem]]}"
            )

            # relax existing exponents for this element (one pass per exponent)
            affected = elem_mask.get(elem, [])
            for k in range(len(basis[elem])):
                l_k, exp_k = basis[elem][k]
                best_exp_k = exp_k
                best_rmse_k = current_rmse
                best_errors_k = error_cache.copy()

                for sign in (+1, -1):
                    for p_abs in range(1, 10_000):
                        trial_exp = exp_k * (STEP_FACTOR ** (sign * p_abs))
                        trial_prims = complete_prims(basis)
                        trial_prims[elem][k] = (l_k, trial_exp)
                        trial_errors = eval_trial(
                            trial_prims, affected, error_cache
                        )
                        rmse = rmse_from_errors(trial_errors)
                        if rmse < best_rmse_k:
                            best_rmse_k = rmse
                            best_exp_k = trial_exp
                            best_errors_k = trial_errors
                        else:
                            break

                if best_exp_k != exp_k:
                    basis[elem][k] = (l_k, best_exp_k)
                    error_cache[:] = best_errors_k
                    current_rmse = best_rmse_k
                    print(
                        f"    relaxed {L_NAMES[l_k]}:"
                        f" {exp_k:.6f} -> {best_exp_k:.6f},"
                        f" RMSE={current_rmse * 1000:.4f} mHa"
                    )

        if not any_added:
            print(f"\nNo functions added in round {round_num}. Stopping.")
            break

    pool.close()
    pool.join()

    total = sum(len(v) for v in basis.values())
    print(f"\nFinal basis: {total} functions, RMSE={current_rmse * 1000:.4f} mHa")
    write_gbs(basis, output_path)
    print(f"Written to: {output_path}")


def cabs_opt_basinhopping(
    atomspecs: list[str],
    cbs_estimates: list[float],
    obs_basis: str,
    output_path: str,
    element: str = "C",
    shell_counts: dict | None = None,
    niter: int = 200,
    T: float = 0.3,
    stepsize: float = 0.2,
):
    """Optimize CABS exponents for one element via basin hopping.

    The shell structure is fixed (default: 3S 3P 2D 1F for C).  Exponents
    are optimised in log-space so they stay positive and can span orders of
    magnitude.  The objective is the stoichiometry-detrended RMSE across
    all molecules; molecules not containing *element* are unaffected and
    contribute their uncorrected E_HF − E_CBS error throughout.

    Progress is printed at every new global best found by the objective and
    at every completed basin-hop step (accepted or rejected).
    """
    L_NAMES = {0: "S", 1: "P", 2: "D", 3: "F"}

    if shell_counts is None:
        shell_counts = {0: 3, 1: 3, 2: 2, 3: 1}

    # ordered list of (l, slot_index) defining the parameter vector
    shell_spec = [(l, k) for l in sorted(shell_counts) for k in range(shell_counts[l])]
    n_params = len(shell_spec)

    def log_exps_to_shells(log_exps):
        return [(l, float(np.exp(log_exps[i]))) for i, (l, _) in enumerate(shell_spec)]

    def primitives_to_pyscf(prims):
        return {
            elem: [[l, [exp, 1.0]] for l, exp in shells]
            for elem, shells in prims.items()
        }

    def write_gbs(shells, path):
        lines = ["****", f"{element} 0"]
        for l, exp in shells:
            lines.append(f"{L_NAMES[l]}   1   1.00")
            lines.append(f"      {exp:>20.10f}   1.0000000000")
        lines.append("****")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # --- setup ---
    all_elements = sorted({e for spec in atomspecs for e in _count_elements(spec)})

    X = np.zeros((len(atomspecs), len(all_elements)))
    for i, spec in enumerate(atomspecs):
        counts = _count_elements(spec)
        for j, elem in enumerate(all_elements):
            X[i, j] = counts.get(elem, 0)

    elem_mask = {}
    for i, atomspec in enumerate(atomspecs):
        for e in _count_elements(atomspec):
            elem_mask.setdefault(e, []).append(i)

    _spawn = mp.get_context("spawn")
    with _spawn.Pool() as hf_pool:
        hf_results = hf_pool.map(
            _hf_fill_worker,
            [(spec, obs_basis) for spec in atomspecs],
        )

    picklable_hf_cache = [
        (spec, F_ao, mo_coeff, E_hf)
        for spec, (E_hf, mo_coeff, F_ao) in zip(atomspecs, hf_results)
    ]
    E_hf_list = [entry[3] for entry in picklable_hf_cache]

    pool = _spawn.Pool(
        initializer=_pool_init,
        initargs=(picklable_hf_cache, obs_basis, list(cbs_estimates)),
    )

    def rmse_from_errors(errors):
        alpha, _, _, _ = np.linalg.lstsq(X, errors, rcond=None)
        residuals = errors - X @ alpha
        return float(np.sqrt(np.mean(residuals**2)))

    def complete_prims(element_shells):
        """Build a full prims dict with empty lists for all uncovered elements."""
        return {elem: (list(element_shells) if elem == element else []) for elem in all_elements}

    # baseline errors with no CABS
    base_errors = np.array(
        [e - c for e, c in zip(E_hf_list, cbs_estimates)], dtype=float
    )
    baseline_rmse = rmse_from_errors(base_errors)
    print(f"Baseline RMSE (no CABS, {element}): {baseline_rmse * 1000:.4f} mHa")
    print(f"Shell structure: {' '.join(f'{n}{L_NAMES[l]}' for l, n in sorted((l,n) for l,n in shell_counts.items()))}")
    print(f"Parameters: {n_params} exponents in log-space")
    print(f"Basin hopping: niter={niter}, T={T}, stepsize={stepsize}\n")

    affected = elem_mask.get(element, [])

    # mutable state for the objective closure
    eval_count = [0]
    best_rmse = [baseline_rmse]
    best_shells = [None]

    def objective(log_exps):
        eval_count[0] += 1
        shells = log_exps_to_shells(log_exps)
        trial_pyscf = primitives_to_pyscf(complete_prims(shells))
        trial_errors = base_errors.copy()
        for i, val in pool.map(
            _worker_compute_one, [(i, trial_pyscf) for i in affected]
        ):
            trial_errors[i] = val
        rmse = rmse_from_errors(trial_errors)

        is_best = rmse < best_rmse[0]
        if is_best:
            best_rmse[0] = rmse
            best_shells[0] = shells

        exp_str = " ".join(f"{np.exp(v):.3f}" for v in log_exps)
        marker = " *** NEW BEST ***" if is_best else ""
        print(
            f"  eval {eval_count[0]:5d}:  {rmse * 1000:.5f} mHa"
            f"  (best {best_rmse[0] * 1000:.5f})  [{exp_str}]{marker}"
        )

        return rmse

    def callback(log_exps, f, accepted):
        exp_str = "  ".join(
            f"{L_NAMES[l]}={np.exp(log_exps[i]):.4f}"
            for i, (l, _) in enumerate(shell_spec)
        )
        status = "ACCEPTED" if accepted else "rejected"
        print(
            f"\n[basin hop] {status}  f={f * 1000:.5f} mHa"
            f"  best={best_rmse[0] * 1000:.5f} mHa"
            f"\n            {exp_str}\n"
        )

    # initial point: one point per angular momentum, spread in log-space,
    # with a slight offset between functions of the same l to break symmetry
    x0 = []
    for l in sorted(shell_counts):
        n = shell_counts[l]
        centers = np.linspace(np.log(0.1), np.log(100.0), n)
        x0.extend(centers.tolist())
    x0 = np.array(x0)

    print(f"Starting exponents: {[f'{np.exp(v):.4f}' for v in x0]}\n")

    result = basinhopping(
        objective,
        x0,
        niter=niter,
        T=T,
        stepsize=stepsize,
        minimizer_kwargs={"method": "Nelder-Mead", "options": {"xatol": 1e-4, "fatol": 1e-8}},
        callback=callback,
        seed=42,
    )

    pool.close()
    pool.join()

    final_shells = log_exps_to_shells(result.x)
    print(f"\n=== Optimization complete ===")
    print(f"Best RMSE: {result.fun * 1000:.5f} mHa  (baseline: {baseline_rmse * 1000:.5f} mHa)")
    print(f"Reduction:   {(1 - result.fun / baseline_rmse) * 100:.1f}%")
    print(f"Evaluations: {eval_count[0]}")
    print("Final exponents:")
    for l, exp in final_shells:
        print(f"  {L_NAMES[l]}: {exp:.6f}")

    write_gbs(final_shells, output_path)
    print(f"Written to: {output_path}")


def cabs_opt_differential_evolution(
    atomspecs: list[str],
    cbs_estimates: list[float],
    obs_basis: str,
    output_path: str,
    element: str = "C",
    shell_counts: dict | None = None,
    niter: int = 200,
    popsize: int = 15,
    mutation: tuple = (0.5, 1.0),
    recombination: float = 0.7,
):
    """Optimise CABS exponents for one element via differential evolution.

    Drop-in alternative to cabs_opt_basinhopping: same first six positional
    arguments and same shell-structure defaults (3S 3P 2D 1F for C).
    Exponents are searched in log-space over [log(0.05), log(500)].
    The objective is the stoichiometry-detrended RMSE across all molecules.

    *niter* sets maxiter (generations), *popsize* the population size factor
    (total population = popsize * n_params), *mutation* the differential
    weight F (scalar or (min, max) for dithering), *recombination* the
    crossover probability CR.
    """
    L_NAMES = {0: "S", 1: "P", 2: "D", 3: "F"}
    LOG_LO = np.log(0.05)
    LOG_HI = np.log(500.0)

    if shell_counts is None:
        shell_counts = {0: 3, 1: 3, 2: 2, 3: 1}

    shell_spec = [(l, k) for l in sorted(shell_counts) for k in range(shell_counts[l])]
    n_params = len(shell_spec)
    bounds = [(LOG_LO, LOG_HI)] * n_params

    def log_exps_to_shells(log_exps):
        return [(l, float(np.exp(log_exps[i]))) for i, (l, _) in enumerate(shell_spec)]

    def primitives_to_pyscf(prims):
        return {
            elem: [[l, [exp, 1.0]] for l, exp in shells]
            for elem, shells in prims.items()
        }

    def write_gbs(shells, path):
        lines = ["****", f"{element} 0"]
        for l, exp in shells:
            lines.append(f"{L_NAMES[l]}   1   1.00")
            lines.append(f"      {exp:>20.10f}   1.0000000000")
        lines.append("****")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # --- setup ---
    all_elements = sorted({e for spec in atomspecs for e in _count_elements(spec)})

    X = np.zeros((len(atomspecs), len(all_elements)))
    for i, spec in enumerate(atomspecs):
        counts = _count_elements(spec)
        for j, elem in enumerate(all_elements):
            X[i, j] = counts.get(elem, 0)

    elem_mask = {}
    for i, atomspec in enumerate(atomspecs):
        for e in _count_elements(atomspec):
            elem_mask.setdefault(e, []).append(i)

    _spawn = mp.get_context("spawn")
    with _spawn.Pool() as hf_pool:
        hf_results = hf_pool.map(
            _hf_fill_worker,
            [(spec, obs_basis) for spec in atomspecs],
        )

    picklable_hf_cache = [
        (spec, F_ao, mo_coeff, E_hf)
        for spec, (E_hf, mo_coeff, F_ao) in zip(atomspecs, hf_results)
    ]
    E_hf_list = [entry[3] for entry in picklable_hf_cache]

    pool = _spawn.Pool(
        initializer=_pool_init,
        initargs=(picklable_hf_cache, obs_basis, list(cbs_estimates)),
    )

    def rmse_from_errors(errors):
        alpha, _, _, _ = np.linalg.lstsq(X, errors, rcond=None)
        residuals = errors - X @ alpha
        return float(np.sqrt(np.mean(residuals**2)))

    def complete_prims(element_shells):
        return {elem: (list(element_shells) if elem == element else []) for elem in all_elements}

    base_errors = np.array(
        [e - c for e, c in zip(E_hf_list, cbs_estimates)], dtype=float
    )
    baseline_rmse = rmse_from_errors(base_errors)
    print(f"Baseline RMSE (no CABS, {element}): {baseline_rmse * 1000:.4f} mHa")
    print(f"Shell structure: {' '.join(f'{n}{L_NAMES[l]}' for l, n in sorted((l, n) for l, n in shell_counts.items()))}")
    print(f"Parameters: {n_params} exponents in log-space, bounds=[{np.exp(LOG_LO):.2f}, {np.exp(LOG_HI):.0f}]")
    print(f"Differential evolution: maxiter={niter}, popsize={popsize}, mutation={mutation}, recombination={recombination}\n")

    affected = elem_mask.get(element, [])
    eval_count = [0]
    best_rmse = [baseline_rmse]
    best_shells = [None]
    generation = [0]
    # collect per-generation RMSE values to print population statistics
    gen_rmse_values = []

    def objective(log_exps):
        eval_count[0] += 1
        shells = log_exps_to_shells(log_exps)
        trial_pyscf = primitives_to_pyscf(complete_prims(shells))
        trial_errors = base_errors.copy()
        for i, val in pool.map(
            _worker_compute_one, [(i, trial_pyscf) for i in affected]
        ):
            trial_errors[i] = val
        rmse = rmse_from_errors(trial_errors)
        gen_rmse_values.append(rmse)

        is_best = rmse < best_rmse[0]
        if is_best:
            best_rmse[0] = rmse
            best_shells[0] = shells

        exp_str = " ".join(f"{np.exp(v):.3f}" for v in log_exps)
        marker = " *** NEW BEST ***" if is_best else ""
        print(
            f"  eval {eval_count[0]:5d} [gen {generation[0]:4d}]:"
            f"  {rmse * 1000:.5f} mHa"
            f"  (best {best_rmse[0] * 1000:.5f})  [{exp_str}]{marker}"
        )
        return rmse

    def callback(xk, convergence):
        generation[0] += 1
        vals = np.array(gen_rmse_values) * 1000  # mHa
        gen_rmse_values.clear()
        print(
            f"\n{'='*70}"
            f"\n[generation {generation[0]:4d}]  convergence={convergence:.6f}"
            f"  evals this gen: {len(vals)}"
            f"\n  population RMSE (mHa):"
            f"  min={vals.min():.5f}  mean={vals.mean():.5f}"
            f"  max={vals.max():.5f}  std={vals.std():.5f}"
            f"\n  global best: {best_rmse[0] * 1000:.5f} mHa"
            f"  ({(1 - best_rmse[0] / baseline_rmse) * 100:.1f}% reduction)"
            f"\n  best exps:   {' '.join(f'{L_NAMES[l]}={np.exp(xk[i]):.5f}' for i, (l, _) in enumerate(shell_spec))}"
            f"\n{'='*70}\n"
        )

    result = differential_evolution(
        objective,
        bounds,
        maxiter=niter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        callback=callback,
        seed=42,
        tol=1e-8,
        polish=False,
    )

    pool.close()
    pool.join()

    final_shells = log_exps_to_shells(result.x)
    print(f"\n=== Optimisation complete ===")
    print(f"Best RMSE:   {result.fun * 1000:.5f} mHa  (baseline: {baseline_rmse * 1000:.5f} mHa)")
    print(f"Reduction:   {(1 - result.fun / baseline_rmse) * 100:.1f}%")
    print(f"Evaluations: {eval_count[0]}")
    print("Final exponents:")
    for l, exp in final_shells:
        print(f"  {L_NAMES[l]}: {exp:.6f}")

    write_gbs(final_shells, output_path)
    print(f"Written to: {output_path}")


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
