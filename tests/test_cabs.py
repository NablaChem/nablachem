from pathlib import Path

import pytest

from nablachem.cabs import CABS_singles_RHF

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
