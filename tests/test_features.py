import pathlib

import numpy as np
import pytest
from ase import Atoms

from nablachem.krr.dataset import DataSet
import nablachem.krr.features as features

pytestmark = pytest.mark.isolated_process

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"

_available = features.list_available()
_rep_class_map = {name: getattr(features, name) for name in _available}
local_reps = [cls for name, cls in _rep_class_map.items() if name.endswith("Local")]
global_reps = [cls for name, cls in _rep_class_map.items() if name.endswith("Global")]


class _FakeDS:
    def __init__(self, molecules):
        self.molecules = molecules


@pytest.fixture
def single_mol_dataset():
    return DataSet(str(DATA_FILE), "A", limit=1)


@pytest.fixture
def mol_h2o():
    return Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


@pytest.fixture
def mol_hcn():
    return Atoms("HCN", positions=[[0, 0, 0], [1.065, 0, 0], [2.221, 0, 0]])


@pytest.fixture
def mol_co2():
    return Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])


def test_all_reps_have_locality_suffix():
    bad = [name for name in _available if not name.endswith(("Local", "Global"))]
    assert not bad, f"Reps without Local/Global suffix: {bad}"


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_is_ndarray(single_mol_dataset, RepClass):
    RepClass().build(single_mol_dataset)
    assert isinstance(single_mol_dataset.representations[0], np.ndarray)


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_is_2d(single_mol_dataset, RepClass):
    RepClass().build(single_mol_dataset)
    assert single_mol_dataset.representations[0].ndim == 2


@pytest.mark.parametrize("RepClass", local_reps, ids=lambda c: c.__name__)
def test_local_rep_natoms(single_mol_dataset, RepClass):
    RepClass().build(single_mol_dataset)
    rep = single_mol_dataset.representations[0]
    assert rep.shape[0] == len(single_mol_dataset.molecules[0])


@pytest.mark.parametrize("RepClass", global_reps, ids=lambda c: c.__name__)
def test_global_rep_is_ndarray(single_mol_dataset, RepClass):
    RepClass().build(single_mol_dataset)
    assert isinstance(single_mol_dataset.representations[0], np.ndarray)


@pytest.mark.parametrize("RepClass", global_reps, ids=lambda c: c.__name__)
def test_global_rep_is_1d(single_mol_dataset, RepClass):
    RepClass().build(single_mol_dataset)
    assert single_mol_dataset.representations[0].ndim == 1


@pytest.mark.parametrize("rep_name", _available)
def test_slice_invariance(rep_name, mol_h2o, mol_hcn, mol_co2):
    """Feature vector for a molecule must be identical regardless of what else is in the slice."""
    rep = _rep_class_map[rep_name]()
    rep.build(_FakeDS([mol_h2o, mol_hcn, mol_co2]))

    alone_h2o = rep[0]
    alone_hcn = rep[1]
    alone_co2 = rep[2]
    together = rep[0:3]

    np.testing.assert_array_equal(alone_h2o, together[0])
    np.testing.assert_array_equal(alone_hcn, together[1])
    np.testing.assert_array_equal(alone_co2, together[2])


class _Rows(features.BaseRepresenter):
    """Local representation with one hand-picked row per atom."""

    def compute(self, molecules: list) -> list:
        return [np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) for _ in molecules]


def test_orbital_weighting_sums_weighted_rows(monkeypatch, mol_h2o):
    monkeypatch.setattr(
        features,
        "_mulliken_weights",
        lambda mol, mo: np.array([0.5, 0.25, 0.25]),
    )
    rep = features._OrbitalWeighted(_Rows(), "HOMO")
    rep.build(_FakeDS([mol_h2o]))

    np.testing.assert_allclose(rep[0], [0.75, 0.5])


def test_orbital_weighting_rejects_global_representation(mol_h2o):
    rep = features._OrbitalWeighted(features.SLATMGlobal(), "HOMO")
    rep.build(_FakeDS([mol_h2o]))

    with pytest.raises(ValueError, match="local representation"):
        rep[0]


def test_mulliken_weights_of_water_sit_on_oxygen(mol_h2o):
    weights = features._mulliken_weights(mol_h2o, "HOMO")

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] + weights[2]  # the HOMO is an oxygen lone pair


@pytest.mark.parametrize("rep_name", _available)
def test_compatible_to_invariance(rep_name, mol_h2o, mol_hcn, mol_co2):
    """Feature vector must be identical whether the other molecule is in compatible_to or the same ds."""
    rep_joint = _rep_class_map[rep_name]()
    rep_joint.build(_FakeDS([mol_h2o, mol_hcn, mol_co2]))

    rep_split = _rep_class_map[rep_name]()
    rep_split.build(
        _FakeDS([mol_h2o]), compatible_to=[_FakeDS([mol_hcn]), _FakeDS([mol_co2])]
    )

    np.testing.assert_array_equal(rep_split[0], rep_joint[0])
