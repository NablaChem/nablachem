import pathlib

import ase
import pytest

from nablachem.krr.dataset import DataSet

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"


@pytest.fixture(scope="module")
def dataset():
    return DataSet(str(DATA_FILE), "A")


def test_length(dataset):
    assert len(dataset) == 2


def test_molecules_are_ase_atoms(dataset):
    for mol in dataset.molecules:
        assert isinstance(mol, ase.Atoms)


def test_molecule_sizes(dataset):
    for mol in dataset.molecules:
        assert len(mol) == 2


def test_labels_single_column(dataset):
    assert set(dataset.labels) == {1.0, 3.0}


def test_labels_expression():
    ds = DataSet(str(DATA_FILE), "A + B")
    assert set(ds.labels) == {3.0, 7.0}
