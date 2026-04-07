#!/usr/bin/env python3
"""CABS exponent optimization via steepest descent + line scan on the Martinez dataset.

Usage:  martinez_cabs_gd.py SHELL_SPEC
  SHELL_SPEC  uppercase string of shell types to use per non-H element, e.g. SPPD
              repeated letters = multiple shells of that type (first N from initial.gbs)
              H uses all shells from initial.gbs regardless

Phase 0 (once):   report null-model detrended RMSE on full dataset (ceiling)
Phase 1 (once):   precompute pyscf HF for 128-molecule sample (non-C/H elements only)
Phase 2 (loop):   steepest descent + line scan, minimising detrended RMSE of
                  (E_HF + E_CABS − E_CBS); alpha refitted at every evaluation

Input:  initial.gbs   – starting CABS exponents
Output: improved.gbs  – updated whenever a new best RMSE is found
"""

import json
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


from nablachem.cabs import (
    _parse_gbs_per_element,
    form_basissets,
    form_cabs_singles,
    form_fock,
)
from pyscf import gto, scf
import multiprocessing as mp


class _MockSCF:
    """Lightweight stand-in for pyscf RHF that carries precomputed arrays."""

    def __init__(self, mol, F_ao, mo_coeff):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self._F_ao = F_ao

    def get_fock(self):
        return self._F_ao


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


# ── Configuration ─────────────────────────────────────────────────────────────
JSONL_PATH  = "Martinez_CABS_DZ_Mar.jsonl"
OBS_BASIS   = "cc-pvdz"
CABS_INIT   = "initial.gbs"
CABS_OUT    = None  # set to f"improved-{shell_spec}.gbs" after argv parsing
N_SAMPLE        = 128           # 4 × 32
EPS_LOG         = np.log(1.05)  # 5 % forward finite-difference step in log-space
LINE_SCAN_STEPS = np.logspace(-3, 0.3, 14)  # step sizes: ~0.001 … ~2.0
RANDOM_SEED     = 42
L_NAMES     = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G"}

# ── Module-level globals for spawned worker processes ─────────────────────────
_G_hf_cache  = []   # list of (atomspec, F_ao, mo_coeff, E_hf)
_G_obs_basis = ""


def _worker_init(hf_cache, obs_basis):
    global _G_hf_cache, _G_obs_basis
    _G_hf_cache  = hf_cache
    _G_obs_basis = obs_basis


def _worker_cabs_singles(args):
    """Return (local_idx, E_CABS_singles) for the given basis dict."""
    local_idx, basis_dict = args
    atomspec, F_ao, mo_coeff, _ = _G_hf_cache[local_idx]
    mol = gto.Mole()
    mol.atom    = atomspec
    mol.basis   = _G_obs_basis
    mol.unit    = "Angstrom"
    mol.verbose = 0
    mol.build()
    mock_mf = _MockSCF(mol, F_ao, mo_coeff)
    mol_ri, C_cabs_ao = form_basissets(mol, _G_obs_basis, basis_dict)
    f, nocc = form_fock(mock_mf, mol_ri, C_cabs_ao)
    return local_idx, form_cabs_singles(f, nocc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def xyz_to_atomspec(xyz: str) -> str:
    return "\n".join(xyz.strip().splitlines()[2:])


def get_elements(xyz: str) -> set:
    return {ln.split()[0] for ln in xyz.strip().splitlines()[2:] if ln.strip()}


def count_elements(atomspec: str) -> dict:
    counts = {}
    for ln in atomspec.strip().splitlines():
        parts = ln.split()
        if parts:
            e = parts[0].capitalize()
            counts[e] = counts.get(e, 0) + 1
    return counts


def extract_params(basis_dict: dict):
    """Return (structure, u_init).
    structure : list of (elem, shell_idx, l)  in sorted-element order
    u_init    : 1-D array of log(exponents)
    """
    structure, exps = [], []
    for elem in sorted(basis_dict):
        for k, shell in enumerate(basis_dict[elem]):
            l   = shell[0]
            exp = float(shell[1][0])
            structure.append((elem, k, l))
            exps.append(exp)
    return structure, np.log(np.array(exps, dtype=float))


def build_basis_dict(structure, u, original_basis: dict) -> dict:
    """Deep-copy original basis and replace exponents with exp(u)."""
    result = deepcopy(original_basis)
    for (elem, k, _l), u_k in zip(structure, u):
        result[elem][k][1][0] = float(np.exp(u_k))
    return result


def write_gbs(basis_dict: dict, path: str):
    lines = []
    for elem in sorted(basis_dict):
        lines.append("****")
        lines.append(f"{elem} 0")
        for shell in basis_dict[elem]:
            l, (exp, coeff) = shell[0], shell[1]
            lines.append(f"{L_NAMES[l]}   1   1.00")
            lines.append(f"      {exp:>20.10f}   1.0000000000")
    lines.append("****")
    Path(path).write_text("\n".join(lines) + "\n")


def rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


NAME_TO_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}


def parse_shell_spec(spec: str) -> dict:
    """Parse e.g. 'SPPD' → {0: 1, 1: 2, 2: 1}  (l → count)."""
    counts = {}
    for ch in spec:
        if ch not in NAME_TO_L:
            raise ValueError(f"Unknown shell type '{ch}' in spec '{spec}' "
                             f"— only S P D F G are allowed")
        l = NAME_TO_L[ch]
        counts[l] = counts.get(l, 0) + 1
    return counts


def apply_shell_spec(basis_dict: dict, shell_counts: dict) -> dict:
    """Return a filtered copy of basis_dict.

    For every non-H element, keep only the first N shells of each angular
    momentum type as specified by shell_counts.  H is passed through unchanged.
    Raises ValueError if an element has fewer shells of a type than required.
    """
    result = {}
    for elem, shells in basis_dict.items():
        if elem == "H":
            result[elem] = list(shells)
            continue

        by_l: dict[int, list] = {}
        for shell in shells:
            by_l.setdefault(shell[0], []).append(shell)

        selected = []
        for l, needed in sorted(shell_counts.items()):
            available = by_l.get(l, [])
            if len(available) < needed:
                raise ValueError(
                    f"Element {elem}: need {needed} {L_NAMES[l]} shell(s) "
                    f"but initial.gbs only has {len(available)}"
                )
            selected.extend(available[:needed])

        selected.sort(key=lambda s: s[0])   # keep angular-momentum order
        result[elem] = selected
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: martinez_cabs_gd.py SHELL_SPEC   (e.g. SPPD)")
        sys.exit(1)
    shell_spec   = sys.argv[1]
    shell_counts = parse_shell_spec(shell_spec)
    global CABS_OUT
    CABS_OUT = f"improved-{shell_spec}.gbs"
    print(f"Shell spec: '{shell_spec}' → "
          + "  ".join(f"{L_NAMES[l]}×{n}" for l, n in sorted(shell_counts.items())))

    random.seed(RANDOM_SEED)

    # ── Phase 0: full-dataset detrending ─────────────────────────────────────
    print("=" * 70)
    print("Phase 0: full-dataset detrending")
    print("=" * 70)

    with open(JSONL_PATH) as fh:
        records = [json.loads(ln) for ln in fh]
    N_full = len(records)

    atomspecs_full = [xyz_to_atomspec(r["xyz"]) for r in records]

    all_elems = sorted({e for spec in atomspecs_full for e in count_elements(spec)})
    X_full = np.zeros((N_full, len(all_elems)))
    for i, spec in enumerate(atomspecs_full):
        for j, e in enumerate(all_elems):
            X_full[i, j] = count_elements(spec).get(e, 0)

    E_hf_full  = np.array([r["RHF/pcseg-0"]       for r in records])
    E_cbs_full = np.array([r["RHF/cc-pv5z"] for r in records])
    null_errors = E_hf_full - E_cbs_full               # positive: small basis > CBS

    alpha, _, _, _ = np.linalg.lstsq(X_full, null_errors, rcond=None)
    targets_full   = null_errors - X_full @ alpha      # fixed residuals per molecule

    print(f"  Molecules : {N_full}")
    print(f"  Elements  : {all_elems}")
    print(f"  alpha (mHa/atom) : "
          + "  ".join(f"{e}={a*1000:.3f}" for e, a in zip(all_elems, alpha)))
    print(f"  Null-model detrended RMSE = {rmse(targets_full)*1000:.4f} mHa\n")

    # ── Phase 1: sample selection ─────────────────────────────────────────────
    print("=" * 70)
    print(f"Phase 1: selecting {N_SAMPLE} sample molecules")
    print("=" * 70)

    niche_pool = [
        (i, r)
        for i, r in enumerate(records)
        if get_elements(r["xyz"]) - {"C", "H"}
    ]

    # stratify by element-combination (beyond C/H)
    bins = defaultdict(list)
    for item in niche_pool:
        key = frozenset(get_elements(item[1]["xyz"]) - {"C", "H"})
        bins[key].append(item)

    selected = []
    for key, items in bins.items():
        n_bin = max(1, round(N_SAMPLE * len(items) / len(niche_pool)))
        selected.extend(random.sample(items, min(n_bin, len(items))))
    random.shuffle(selected)
    selected = selected[:N_SAMPLE]

    sample_specs = [xyz_to_atomspec(r["xyz"]) for _, r in selected]

    # Precompute fixed per-sample quantities (no QM needed)
    null_error_sample = np.array([r["RHF/cc-pvdz"] - r["RHF/cc-pv5z"] for _, r in selected])

    all_elems_sample = sorted({e for spec in sample_specs for e in count_elements(spec)})
    X_sample = np.zeros((N_SAMPLE, len(all_elems_sample)))
    for i, spec in enumerate(sample_specs):
        for j, e in enumerate(all_elems_sample):
            X_sample[i, j] = count_elements(spec).get(e, 0)

    # elem_mask: elem → list of local indices in sample
    elem_to_local = defaultdict(list)
    for local_idx, spec in enumerate(sample_specs):
        for e in count_elements(spec):
            elem_to_local[e].append(local_idx)

    print(f"  Niche pool : {len(niche_pool)}")
    for key, items in sorted(bins.items()):
        print(f"    {set(key)} : {len(items)} molecules")
    print(f"  Sample     : {len(selected)} molecules")
    print(f"  Element coverage in sample:")
    for e, idxs in sorted(elem_to_local.items()):
        print(f"    {e} : {len(idxs)} molecules")

    # Pre-compute HF for sample molecules
    n_cores = mp.cpu_count()
    print(f"\n  Pre-computing HF/{OBS_BASIS} on {n_cores} cores …")
    _spawn = mp.get_context("spawn")
    t0 = time.time()
    with _spawn.Pool() as hf_pool:
        hf_results = hf_pool.map(
            _hf_fill_worker,
            [(spec, OBS_BASIS) for spec in sample_specs],
        )
    print(f"  HF done in {time.time()-t0:.1f}s")

    # Verify stored pcseg vs computed
    stored_hf   = np.array([r["RHF/pcseg-0"] for _, r in selected])
    computed_hf = np.array([res[0] for res in hf_results])
    print(f"  Max |stored pcseg − computed HF| = "
          f"{np.max(np.abs(stored_hf - computed_hf))*1000:.4f} mHa")

    # (atomspec, F_ao, mo_coeff, E_hf)  — note _hf_fill_worker returns (E_hf, mo_coeff, F_ao)
    hf_cache = [
        (spec, F_ao, mo_coeff, E_hf)
        for spec, (E_hf, mo_coeff, F_ao) in zip(sample_specs, hf_results)
    ]

    # ── Load initial CABS basis ───────────────────────────────────────────────
    print(f"\n  Loading initial CABS from {CABS_INIT} …")
    initial_basis = apply_shell_spec(_parse_gbs_per_element(CABS_INIT), shell_counts)
    structure, u_init = extract_params(initial_basis)
    n_params = len(u_init)

    print(f"  Parameters ({n_params}):")
    for (elem, k, l), u_k in zip(structure, u_init):
        print(f"    {elem} {L_NAMES[l]}[{k}] : exp = {np.exp(u_k):.6f}")

    param_to_elem = [elem for (elem, k, l) in structure]

    # ── Optimisation pool ─────────────────────────────────────────────────────
    pool = _spawn.Pool(
        initializer=_worker_init,
        initargs=(hf_cache, OBS_BASIS),
    )

    # ── Caches to avoid redundant QM ─────────────────────────────────────────
    # _f_cache  : u.tobytes() → float  (cheap: base CABS only)
    # _fg_cache : u.tobytes() → (float, ndarray)  (expensive: base + FD grad)
    _f_cache  = {}
    _fg_cache = {}

    best_rmse  = [np.inf]
    grad_count = [0]
    step_count = [0]

    def _detrended_rmse(e_cabs: np.ndarray) -> float:
        """Refit alpha to (E_HF + E_CABS − E_CBS) and return detrended RMSE."""
        total = null_error_sample + e_cabs
        alpha_fit, _, _, _ = np.linalg.lstsq(X_sample, total, rcond=None)
        return rmse(total - X_sample @ alpha_fit)

    def _eval_base(u) -> tuple:
        """Compute base CABS for all sample molecules; return (e_cabs, detrended_rmse)."""
        basis = build_basis_dict(structure, u, initial_basis)
        results = pool.map(
            _worker_cabs_singles,
            [(li, basis) for li in range(N_SAMPLE)],
        )
        e_cabs = np.empty(N_SAMPLE)
        for li, val in results:
            e_cabs[li] = val
        return e_cabs, _detrended_rmse(e_cabs)

    def _maybe_save_best(u, f):
        if f < best_rmse[0]:
            best_rmse[0] = f
            write_gbs(build_basis_dict(structure, u, initial_basis), CABS_OUT)
            print(f"   → new best {f*1000:.4f} mHa — written to {CABS_OUT}")

    def fun(u):
        key = u.tobytes()
        if key in _fg_cache:
            f = _fg_cache[key][0]
        elif key in _f_cache:
            f = _f_cache[key]
        else:
            _, f = _eval_base(u)
            _f_cache[key] = f
        _maybe_save_best(u, f)
        return f

    def jac(u):
        key = u.tobytes()
        if key in _fg_cache:
            return _fg_cache[key][1]

        t0 = time.time()
        grad_count[0] += 1

        # Base evaluation (always need the e_cabs vector for FD reconstruction)
        e_cabs_base, f_base = _eval_base(u)
        _f_cache[key] = f_base

        # Forward finite differences — batch ALL perturbations in one pool.map
        pert_jobs = []   # (local_idx, basis_dict_for_param_p)
        pert_meta = []   # (param_idx, local_idx)

        for p in range(n_params):
            u_pert = u.copy()
            u_pert[p] += EPS_LOG
            basis_pert = build_basis_dict(structure, u_pert, initial_basis)
            for li in elem_to_local.get(param_to_elem[p], []):
                pert_jobs.append((li, basis_pert))
                pert_meta.append((p, li))

        pert_results = pool.map(_worker_cabs_singles, pert_jobs)

        # Reconstruct perturbed objectives using elem_mask
        pert_e_cabs = {p: e_cabs_base.copy() for p in range(n_params)}
        for (p, li), (_, val) in zip(pert_meta, pert_results):
            pert_e_cabs[p][li] = val

        grad = np.empty(n_params)
        for p in range(n_params):
            f_pert = _detrended_rmse(pert_e_cabs[p])
            grad[p] = (f_pert - f_base) / EPS_LOG

        _fg_cache[key] = (f_base, grad)
        _maybe_save_best(u, f_base)

        elapsed = time.time() - t0
        print(f"\n── Grad eval {grad_count[0]}  RMSE = {f_base*1000:.4f} mHa"
              f"  |∇| = {np.linalg.norm(grad):.5f}  [{elapsed:.1f}s]")
        for (elem, k, l), u_k, g_k in zip(structure, u, grad):
            print(f"   {elem} {L_NAMES[l]}[{k}]  exp={np.exp(u_k):.6f}  "
                  f"grad={g_k:+.5f}")

        return grad

    # ── Run: steepest descent with line scan ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: steepest descent + line scan  (Ctrl-C to stop and keep best)")
    print("=" * 70)
    print(f"  OBS={OBS_BASIS}  init={CABS_INIT}  out={CABS_OUT}")
    print(f"  n_sample={N_SAMPLE}  n_params={n_params}  ε={np.exp(EPS_LOG)-1:.0%}")
    print(f"  line scan steps: {len(LINE_SCAN_STEPS)}"
          f"  ({LINE_SCAN_STEPS[0]:.3f} … {LINE_SCAN_STEPS[-1]:.3f})")
    print(f"  Null-model RMSE ceiling = {rmse(targets_full)*1000:.4f} mHa")

    u = u_init.copy()

    try:
        while step_count[0] < 100:
            # 1. Gradient at current point
            g = jac(u)                          # prints grad report, saves best
            direction = -g / np.linalg.norm(g)  # unit steepest-descent direction

            # 2. Line scan along direction
            print(f"\n── Line scan ({len(LINE_SCAN_STEPS)} steps) ──")
            f_current = _fg_cache[u.tobytes()][0]
            best_t, best_f, best_u = 0.0, f_current, u.copy()

            for t in LINE_SCAN_STEPS:
                u_trial = u + t * direction
                f_trial = fun(u_trial)          # cheap: base CABS only, saves best
                marker  = " ←" if f_trial < best_f else ""
                print(f"   t={t:.4f}  RMSE={f_trial*1000:.4f} mHa{marker}")
                if f_trial < best_f:
                    best_f, best_t, best_u = f_trial, t, u_trial.copy()

            step_count[0] += 1
            print(f"── Step {step_count[0]} done: "
                  f"t*={best_t:.4f}  RMSE {f_current*1000:.4f} → {best_f*1000:.4f} mHa")
            u = best_u

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pool.close()
        pool.join()

    print(f"\nBest RMSE = {best_rmse[0]*1000:.4f} mHa — saved to {CABS_OUT}")


if __name__ == "__main__":
    main()
