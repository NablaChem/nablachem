import re
from pathlib import Path

import pytest
from pyscf import gto

from nablachem.cabs import CABS_singles_RHF, resolve_basis

CABS_DIR = Path(__file__).parent / "cabs"
CABS_GBS = str(CABS_DIR / "pcseg-cabs.gbs")
OBS_BASIS = "pcseg-0"


def _parse_atomspec(run_inp):
    """Extract a PySCF-style atomspec from a Psi4 run.inp's `molecule mol {...}` block."""
    text = Path(run_inp).read_text()
    body = re.search(r"molecule mol \{(.*?)\}", text, re.S).group(1)
    atom_lines = []
    for line in body.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("symmetry") or re.match(r"^-?\d+\s+\d+$", line):
            continue
        atom_lines.append(" ".join(line.split()))
    return "; ".join(atom_lines)


def _parse_ref_energies(run_out):
    """Extract RHF reference energy and CABS singles correction from a Psi4 run.out."""
    text = Path(run_out).read_text()
    ref_hf = float(re.search(r"RHF Reference Energy:\s*(-?\d+\.\d+)", text).group(1))
    ref_singles = float(
        re.search(r"CABS Singles Correction:\s*(-?\d+\.\d+)", text).group(1)
    )
    return ref_hf, ref_singles


def _load_cabs_cases(names):
    """Build parametrize cases (atomspec, ref_hf, ref_singles) from cabs/<name>/run.{inp,out}."""
    cases = []
    for name in names:
        case_dir = CABS_DIR / name
        atomspec = _parse_atomspec(case_dir / "run.inp")
        ref_hf, ref_singles = _parse_ref_energies(case_dir / "run.out")
        cases.append(pytest.param(atomspec, ref_hf, ref_singles, id=name))
    return cases


# Reference values from Psi4 1.10 DF-MP2-F12 (pcseg-0 / pcseg-cabs)
# Geometries in Angstrom, energies in Hartree
@pytest.mark.parametrize(
    "atomspec,ref_hf,ref_singles",
    _load_cabs_cases(["H2O", "CO", "N2"]),
)
def test_cabs_singles_rhf(atomspec, ref_hf, ref_singles):
    e_hf, e_singles = CABS_singles_RHF(atomspec, OBS_BASIS, CABS_GBS, density_fit=False)
    assert (
        abs(e_hf - ref_hf) < 1e-8
    ), f"E_HF mismatch: got {e_hf:.12f}, ref {ref_hf:.12f}, diff {e_hf - ref_hf:.2e}"
    assert (
        abs(e_singles - ref_singles) < 5e-5
    ), f"E_singles mismatch: got {e_singles:.12f}, ref {ref_singles:.12f}, diff {e_singles - ref_singles:.2e}"


def test_dz_alchemy_co_to_n2():
    """Alchemically turning CO into N2 via dZ must reproduce N2 computed directly
    in the (asymmetric) CO basis at the same geometry.

    Starting from the *asymmetric* CO reference (O basis on atom 1, C basis on
    atom 2) rather than a symmetric setup catches bugs where dZ is applied to
    the wrong atom or where only symmetric perturbations happen to work.
    """
    geom = "0 0 0", "0 0 1.1"

    # LHS: CO with dZ raising both nuclei to Z=7 (O: 8 -> 7 is -1, C: 6 -> 7 is +1)
    co_atom = f"O {geom[0]}; C {geom[1]}"
    e_hf_dz, e_singles_dz = CABS_singles_RHF(
        co_atom, OBS_BASIS, CABS_GBS, dZ=(-1, +1)
    )

    # RHS: genuine N2 (Z=7,7) but in CO's asymmetric basis: O functions on the
    # atom-1 center, C functions on the atom-2 center (OBS and CABS alike).
    cabs = resolve_basis(CABS_GBS)
    ref_atom = f"N1 {geom[0]}; N2 {geom[1]}"
    ref_obs = {"N1": gto.basis.load(OBS_BASIS, "O"), "N2": gto.basis.load(OBS_BASIS, "C")}
    ref_cabs = {"N1": cabs["O"], "N2": cabs["C"]}
    e_hf_ref, e_singles_ref = CABS_singles_RHF(ref_atom, ref_obs, ref_cabs)

    assert abs(e_hf_dz - e_hf_ref) < 1e-8, (
        f"E_HF mismatch: dZ {e_hf_dz:.12f} vs ref {e_hf_ref:.12f}, "
        f"diff {e_hf_dz - e_hf_ref:.2e}"
    )
    # derived from SCF eigenvalues on both sides, so limited by SCF conv noise
    assert abs(e_singles_dz - e_singles_ref) < 1e-6, (
        f"E_singles mismatch: dZ {e_singles_dz:.12f} vs ref {e_singles_ref:.12f}, "
        f"diff {e_singles_dz - e_singles_ref:.2e}"
    )


def test_dz_wrong_shape():
    co_atom = "O 0 0 0; C 0 0 1.1"
    with pytest.raises(ValueError, match="dZ must have one entry per atom"):
        CABS_singles_RHF(co_atom, OBS_BASIS, CABS_GBS, dZ=(-1,))


def test_larger():
    case_dir = CABS_DIR / "benzene"
    benzene = _parse_atomspec(case_dir / "run.inp")
    ref_hf, ref_singles = _parse_ref_energies(case_dir / "run.out")
    e_hf, e_singles = CABS_singles_RHF(benzene, OBS_BASIS, CABS_GBS, density_fit=True)
    assert (
        abs(e_hf - ref_hf) < 1e-8
    ), f"E_HF mismatch: got {e_hf:.12f}, ref {ref_hf:.12f}, diff {e_hf - ref_hf:.2e}"
    assert (
        abs(e_singles - ref_singles) < 5e-4
    ), f"E_singles mismatch: got {e_singles:.12f}, ref {ref_singles:.12f}, diff {e_singles - ref_singles:.2e}"
