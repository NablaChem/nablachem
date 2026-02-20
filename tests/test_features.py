import pathlib

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet
from nablachem.krr.features import MBDFLocal, cMBDFLocal, SLATMLocal

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"

LOCAL_REPRESENTERS = [MBDFLocal, cMBDFLocal, SLATMLocal]


@pytest.fixture
def single_mol_dataset():
    return DataSet(str(DATA_FILE), "A", limit=1)


@pytest.mark.parametrize("RepClass", LOCAL_REPRESENTERS, ids=lambda c: c.__name__)
def test_representation_is_ndarray(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert isinstance(single_mol_dataset.representations[0], np.ndarray)


@pytest.mark.parametrize("RepClass", LOCAL_REPRESENTERS, ids=lambda c: c.__name__)
def test_representation_is_2d(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert single_mol_dataset.representations[0].ndim == 2


@pytest.mark.parametrize("RepClass", LOCAL_REPRESENTERS, ids=lambda c: c.__name__)
def test_representation_natoms(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    rep = single_mol_dataset.representations[0]
    assert rep.shape[0] == len(single_mol_dataset.molecules[0])
