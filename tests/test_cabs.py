from pathlib import Path

import pytest
from pyscf import gto

from nablachem.cabs import CABS_singles_RHF, resolve_basis

CABS_GBS = str(Path(__file__).parent / "cabs" / "pcseg-cabs.gbs")
OBS_BASIS = "pcseg-0"


# Reference values from Psi4 1.10 DF-MP2-F12 (pcseg-0 / pcseg-cabs)
# Geometries in Angstrom, energies in Hartree
@pytest.mark.parametrize(
    "atomspec,ref_hf,ref_singles",
    [
        (
            "O 0 0 0.117176; H 0 0.757329 -0.468706; H 0 -0.757329 -0.468706",
            -75.774254726349,
            -0.122825268378,
        ),
        (
            "O 0 0 0; C 0 0 1.1",
            -112.384028820127,
            -0.197290223267,
        ),
        (
            "N 0 0 0; N 0 0 1.1",
            -108.594545355417,
            -0.192662990417,
        ),
    ],
    ids=["H2O", "CO", "N2"],
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
    benzene = """C        0.303     -1.351      0.000                 
C        1.322     -0.413      0.000                 
C        1.019      0.938      0.000                 
C       -0.303      1.351      0.000                 
C       -1.322      0.414      0.000                 
C       -1.019     -0.938      0.000                 
H        0.539     -2.405      0.000                 
H        2.353     -0.737      0.000                 
H        1.813      1.670      0.000                 
H       -0.538      2.405      0.000                 
H       -2.353      0.736      0.000                 
H       -1.814     -1.669      0.000                    
"""
    e_hf, e_singles = CABS_singles_RHF(benzene, OBS_BASIS, CABS_GBS, density_fit=True)
    ref_singles = -0.299603647496
    ref_hf = -230.070214467341
    assert (
        abs(e_hf - ref_hf) < 1e-8
    ), f"E_HF mismatch: got {e_hf:.12f}, ref {ref_hf:.12f}, diff {e_hf - ref_hf:.2e}"
    assert (
        abs(e_singles - ref_singles) < 5e-4
    ), f"E_singles mismatch: got {e_singles:.12f}, ref {ref_singles:.12f}, diff {e_singles - ref_singles:.2e}"
