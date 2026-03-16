import pathlib

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet
import nablachem.krr.features as features

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"

_available = features.list_available()
_rep_class_map = {name: getattr(features, name) for name in _available}
local_reps = [cls for name, cls in _rep_class_map.items() if name.endswith("Local")]
global_reps = [cls for name, cls in _rep_class_map.items() if name.endswith("Global")]


@pytest.fixture
def single_mol_dataset():
    return DataSet(str(DATA_FILE), "A", limit=1)


def test_all_reps_have_locality_suffix():
    bad = [name for name in _available if not name.endswith(("Local", "Global"))]
    assert not bad, f"Reps without Local/Global suffix: {bad}"


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_is_ndarray(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert isinstance(single_mol_dataset.representations[0], np.ndarray)


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_is_2d(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert single_mol_dataset.representations[0].ndim == 2


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_natoms(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    rep = single_mol_dataset.representations[0]
    assert rep.shape[0] == len(single_mol_dataset.molecules[0])


@pytest.mark.parametrize("RepClass", global_reps, ids=lambda c: c.__name__)
def test_global_rep_is_ndarray(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert isinstance(single_mol_dataset.representations[0], np.ndarray)


@pytest.mark.parametrize("RepClass", global_reps, ids=lambda c: c.__name__)
def test_global_rep_is_1d(single_mol_dataset, RepClass):
    RepClass().build([single_mol_dataset])
    assert single_mol_dataset.representations[0].ndim == 1
