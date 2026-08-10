import gzip
import json
import pathlib
import re

import ase
import numpy as np
import pytest

from nablachem.krr.dataset import DataSet

# --- existing tests ---

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
        assert len(mol) == 3


def test_labels_single_column(dataset):
    assert set(dataset.labels) == {1.0, 3.0}


def test_labels_expression():
    ds = DataSet(str(DATA_FILE), "A + B")
    assert set(ds.labels) == {3.0, 7.0}


# --- test data for reservoir / batch tests ---
#
# 25 records in three groups:
#   H2O  (idx  0- 9): n_H=2, n_O=1, n_atoms=3, energy = -1 .. -10
#   CO2  (idx 10-19): n_C=1, n_O=2, n_atoms=3, energy =  1 ..  10
#   CH4  (idx 20-24): n_C=1, n_H=4, n_atoms=5, energy = -101 .. -105
#
# Useful counts for assertions:
#   N_POSITIVE_ENERGY = 10  (CO2 only,       Mode B: energy > 0)
#   N_WITH_HYDROGEN   = 15  (H2O + CH4,      Mode C: n_H > 0)
#   N_WITH_CARBON     = 15  (CO2 + CH4,      Mode C: n_C > 0)
#   N_LARGE_MOLECULES =  5  (CH4 only,       Mode C: n_atoms > 3)

XYZ_H2O = "3\nH2O\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469"
XYZ_CO2 = "3\nCO2\nC 0.000 0.000 0.000\nO 0.000 0.000 1.160\nO 0.000 0.000 -1.160"
XYZ_CH4 = "5\nCH4\nC 0.000 0.000 0.000\nH 0.000 1.089 0.000\nH 1.026 -0.363 0.000\nH -0.513 -0.363 0.890\nH -0.513 -0.363 -0.890"

N_TOTAL = 25
N_H2O = 10
N_CO2 = 10
N_CH4 = 5

N_POSITIVE_ENERGY = N_CO2
N_WITH_HYDROGEN = N_H2O + N_CH4
N_WITH_CARBON = N_CO2 + N_CH4
N_LARGE_MOLECULES = N_CH4

IDX_H2O = set(range(0, 10))
IDX_CO2 = set(range(10, 20))
IDX_CH4 = set(range(20, 25))
IDX_WITH_HYDROGEN = IDX_H2O | IDX_CH4
IDX_WITH_CARBON = IDX_CO2 | IDX_CH4

# 7 rows per batch → ceil(25/7) = 4 batches, guarantees multi-batch behaviour
BATCH_SIZE = 7

_CALC_COL_RE = re.compile(r"\bn_atoms\b|\bn_[A-Z][a-z]?\b")


def make_records():
    records = []
    for i in range(N_H2O):
        records.append({"xyz": XYZ_H2O, "energy": -(i + 1), "idx": len(records)})
    for i in range(N_CO2):
        records.append({"xyz": XYZ_CO2, "energy": i + 1, "idx": len(records)})
    for i in range(N_CH4):
        records.append({"xyz": XYZ_CH4, "energy": -(i + 101), "idx": len(records)})
    return records


@pytest.fixture
def jsonl_file(tmp_path):
    path = tmp_path / "molecules.jsonl"
    with open(path, "w") as f:
        for r in make_records():
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def jsonl_gz_file(tmp_path):
    path = tmp_path / "molecules.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for r in make_records():
            f.write(json.dumps(r) + "\n")
    return str(path)


# --- _detect_select_mode ---


def test_detect_mode_none():
    assert DataSet._detect_select_mode(None) == "A"


def test_detect_mode_native_single_column():
    assert DataSet._detect_select_mode("energy > 0") == "B"


def test_detect_mode_native_compound_expression():
    assert DataSet._detect_select_mode("energy > -5 and energy < 5") == "B"


def test_detect_mode_n_atoms():
    assert DataSet._detect_select_mode("n_atoms > 3") == "C"


def test_detect_mode_n_element_one_letter():
    assert DataSet._detect_select_mode("n_H > 0") == "C"


def test_detect_mode_n_element_two_letters():
    assert DataSet._detect_select_mode("n_Cl > 0") == "C"


def test_detect_mode_mixed_native_and_calc():
    assert DataSet._detect_select_mode("n_C > 0 and energy > 0") == "C"


def test_detect_mode_word_boundary_not_triggered():
    # 'neutral_energy' contains 'n_' but the word-boundary anchor must prevent a match
    assert DataSet._detect_select_mode("neutral_energy > 0") == "B"


# --- _reservoir_sample: Mode A (no select) ---


def test_reservoir_mode_a_limit_less_than_total(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 10, None, batch_size=BATCH_SIZE)
    assert len(df) == 10


def test_reservoir_mode_a_limit_equals_total(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, N_TOTAL, None, batch_size=BATCH_SIZE)
    assert len(df) == N_TOTAL


def test_reservoir_mode_a_limit_more_than_total(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_TOTAL + 100, None, batch_size=BATCH_SIZE
    )
    assert len(df) == N_TOTAL


def test_reservoir_mode_a_result_has_xyz_column(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 5, None, batch_size=BATCH_SIZE)
    assert "xyz" in df.columns


def test_reservoir_mode_a_rows_are_valid_json_records(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 10, None, batch_size=BATCH_SIZE)
    assert set(df.columns) >= {"xyz", "energy", "idx"}


def test_reservoir_mode_a_gzip(jsonl_gz_file):
    df = DataSet._reservoir_sample(jsonl_gz_file, 10, None, batch_size=BATCH_SIZE)
    assert len(df) == 10
    assert "xyz" in df.columns


def test_reservoir_mode_a_idxs_subset_of_total(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 15, None, batch_size=BATCH_SIZE)
    assert set(df["idx"]).issubset(set(range(N_TOTAL)))


def test_reservoir_mode_a_no_duplicate_rows(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, N_TOTAL, None, batch_size=BATCH_SIZE)
    assert df["idx"].nunique() == N_TOTAL


# --- _reservoir_sample: Mode B (select on native columns) ---


def test_reservoir_mode_b_limit_less_than_matching(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 5, "energy > 0", batch_size=BATCH_SIZE)
    assert len(df) == 5


def test_reservoir_mode_b_limit_equals_matching(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_POSITIVE_ENERGY, "energy > 0", batch_size=BATCH_SIZE
    )
    assert len(df) == N_POSITIVE_ENERGY


def test_reservoir_mode_b_limit_more_than_matching(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_TOTAL, "energy > 0", batch_size=BATCH_SIZE
    )
    assert len(df) == N_POSITIVE_ENERGY


def test_reservoir_mode_b_filter_respected(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 8, "energy > 0", batch_size=BATCH_SIZE)
    assert (df["energy"] > 0).all()


def test_reservoir_mode_b_all_matching_rows_reachable(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_POSITIVE_ENERGY, "energy > 0", batch_size=BATCH_SIZE
    )
    assert set(df["idx"]) == IDX_CO2


def test_reservoir_mode_b_no_atom_cols_in_result(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 5, "energy > 0", batch_size=BATCH_SIZE)
    for col in df.columns:
        assert not _CALC_COL_RE.match(col)


# --- _reservoir_sample: Mode C (select on calculated columns) ---


def test_reservoir_mode_c_limit_less_than_matching(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 5, "n_H > 0", batch_size=BATCH_SIZE)
    assert len(df) == 5


def test_reservoir_mode_c_limit_equals_matching(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_WITH_HYDROGEN, "n_H > 0", batch_size=BATCH_SIZE
    )
    assert len(df) == N_WITH_HYDROGEN


def test_reservoir_mode_c_limit_more_than_matching(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_TOTAL, "n_H > 0", batch_size=BATCH_SIZE
    )
    assert len(df) == N_WITH_HYDROGEN


def test_reservoir_mode_c_filter_respected_n_H(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_WITH_HYDROGEN, "n_H > 0", batch_size=BATCH_SIZE
    )
    assert set(df["idx"]) == IDX_WITH_HYDROGEN


def test_reservoir_mode_c_filter_respected_n_C(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_WITH_CARBON, "n_C > 0", batch_size=BATCH_SIZE
    )
    assert set(df["idx"]) == IDX_WITH_CARBON


def test_reservoir_mode_c_large_molecules(jsonl_file):
    df = DataSet._reservoir_sample(
        jsonl_file, N_LARGE_MOLECULES, "n_atoms > 3", batch_size=BATCH_SIZE
    )
    assert len(df) == N_LARGE_MOLECULES
    assert set(df["idx"]) == IDX_CH4


def test_reservoir_mode_c_no_calc_cols_in_result(jsonl_file):
    df = DataSet._reservoir_sample(jsonl_file, 5, "n_C > 0", batch_size=BATCH_SIZE)
    for col in df.columns:
        assert not _CALC_COL_RE.match(col)


def test_reservoir_mode_c_combined_filter(jsonl_file):
    # CO2 and CH4 both have carbon, but only CO2 has energy > 0
    df = DataSet._reservoir_sample(
        jsonl_file, N_TOTAL, "n_C > 0 and energy > 0", batch_size=BATCH_SIZE
    )
    assert set(df["idx"]) == IDX_CO2


# --- DataSet end-to-end: Mode D (no limit) ---


def test_dataset_mode_d_no_select_full_load(jsonl_file):
    ds = DataSet(jsonl_file, "energy")
    assert len(ds) == N_TOTAL


def test_dataset_mode_d_select_b_count(jsonl_file):
    ds = DataSet(jsonl_file, "energy", select="energy > 0")
    assert len(ds) == N_POSITIVE_ENERGY


def test_dataset_mode_d_select_b_labels_positive(jsonl_file):
    ds = DataSet(jsonl_file, "energy", select="energy > 0")
    assert (ds.labels > 0).all()


def test_dataset_mode_d_select_b_correct_label_values(jsonl_file):
    ds = DataSet(jsonl_file, "energy", select="energy > 0")
    assert set(ds.labels) == set(range(1, 11))


def test_dataset_mode_d_select_c_count(jsonl_file):
    ds = DataSet(jsonl_file, "energy", select="n_H > 0")
    assert len(ds) == N_WITH_HYDROGEN


def test_dataset_mode_d_select_c_molecules_have_hydrogen(jsonl_file):
    ds = DataSet(jsonl_file, "energy", select="n_H > 0")
    for mol in ds.molecules:
        assert 1 in mol.get_atomic_numbers()


def test_dataset_mode_d_shuffled(jsonl_file):
    label_sequences = [tuple(DataSet(jsonl_file, "energy").labels) for _ in range(5)]
    assert len(set(label_sequences)) > 1


# --- DataSet end-to-end: Mode A (no select, with limit) ---


def test_dataset_mode_a_limit_less_than_total(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=10)
    assert len(ds) == 10


def test_dataset_mode_a_limit_more_than_total(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=N_TOTAL + 100)
    assert len(ds) == N_TOTAL


def test_dataset_mode_a_molecules_are_ase(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=5)
    for mol in ds.molecules:
        assert isinstance(mol, ase.Atoms)


def test_dataset_mode_a_shuffled(jsonl_file):
    label_sequences = [
        tuple(DataSet(jsonl_file, "energy", limit=20).labels) for _ in range(5)
    ]
    assert len(set(label_sequences)) > 1


# --- DataSet end-to-end: Mode B (select on native cols, with limit) ---


def test_dataset_mode_b_limit_less_than_matching(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=5, select="energy > 0")
    assert len(ds) == 5


def test_dataset_mode_b_limit_more_than_matching(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=N_TOTAL, select="energy > 0")
    assert len(ds) == N_POSITIVE_ENERGY


def test_dataset_mode_b_filter_respected(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=5, select="energy > 0")
    assert (ds.labels > 0).all()


# --- DataSet end-to-end: Mode C (select on calc cols, with limit) ---


def test_dataset_mode_c_limit_less_than_matching(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=5, select="n_H > 0")
    assert len(ds) == 5


def test_dataset_mode_c_limit_more_than_matching(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=N_TOTAL, select="n_H > 0")
    assert len(ds) == N_WITH_HYDROGEN


def test_dataset_mode_c_molecules_have_hydrogen(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=5, select="n_H > 0")
    for mol in ds.molecules:
        assert 1 in mol.get_atomic_numbers()


def test_dataset_mode_c_large_molecules_count(jsonl_file):
    ds = DataSet(jsonl_file, "energy", limit=N_TOTAL, select="n_atoms > 3")
    assert len(ds) == N_LARGE_MOLECULES


# --- gzip support ---


def test_dataset_gzip_full_load(jsonl_gz_file):
    ds = DataSet(jsonl_gz_file, "energy")
    assert len(ds) == N_TOTAL


def test_dataset_gzip_with_limit(jsonl_gz_file):
    ds = DataSet(jsonl_gz_file, "energy", limit=10)
    assert len(ds) == 10


def test_dataset_gzip_with_select_b(jsonl_gz_file):
    ds = DataSet(jsonl_gz_file, "energy", select="energy > 0")
    assert len(ds) == N_POSITIVE_ENERGY


def test_dataset_gzip_with_select_c(jsonl_gz_file):
    ds = DataSet(jsonl_gz_file, "energy", limit=N_TOTAL, select="n_H > 0")
    assert len(ds) == N_WITH_HYDROGEN


# --- reservoir sampling uniformity ---


def test_reservoir_all_items_reachable(jsonl_file):
    # Over many draws, every idx should appear at least once
    seen = set()
    for _ in range(50):
        df = DataSet._reservoir_sample(jsonl_file, 10, None, batch_size=BATCH_SIZE)
        seen.update(df["idx"].tolist())
    assert seen == set(range(N_TOTAL))


def test_reservoir_sampling_is_random(jsonl_file):
    # Five independent samples of size 15 should not all be identical sets
    samples = [
        frozenset(
            DataSet._reservoir_sample(jsonl_file, 15, None, batch_size=BATCH_SIZE)[
                "idx"
            ].tolist()
        )
        for _ in range(5)
    ]
    assert len(set(samples)) > 1


def test_reservoir_approximate_uniformity(jsonl_file):
    # Each of the 25 items should appear in roughly 10/25 = 40% of 300 draws.
    # Expected count: 120.  Allow ±50% → [60, 180] (> 6 sigma, negligible false-fail rate).
    n_trials = 300
    limit = 10
    counts = {i: 0 for i in range(N_TOTAL)}
    for _ in range(n_trials):
        df = DataSet._reservoir_sample(jsonl_file, limit, None, batch_size=BATCH_SIZE)
        for idx in df["idx"]:
            counts[idx] += 1
    expected = n_trials * limit / N_TOTAL
    for idx, count in counts.items():
        assert (
            expected * 0.5 <= count <= expected * 1.5
        ), f"idx {idx}: count {count}, expected ~{expected:.0f}"


# --- charge and spin multiplicity ---


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def charged_jsonl(tmp_path):
    # `energy` mirrors `charge` so that a shuffle which broke the
    # molecule<->charge pairing shows up as an elementwise mismatch
    records = [
        {"xyz": XYZ_H2O, "charge": q, "spin_multiplicity": abs(q) + 1, "energy": q}
        for q in (-2, -1, 0, 1, 2)
    ]
    return _write_jsonl(tmp_path / "charged.jsonl", records)


def test_charge_attached_to_mol_info(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert all("charge" in mol.info for mol in ds.molecules)


def test_spin_attached_to_mol_info(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert all("spin_multiplicity" in mol.info for mol in ds.molecules)


def test_total_charges_values(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert sorted(ds.total_charges) == [-2, -1, 0, 1, 2]


def test_spin_multiplicities_values(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert sorted(ds.spin_multiplicities) == [1, 2, 2, 3, 3]


def test_total_charges_is_integer_dtype(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert np.issubdtype(ds.total_charges.dtype, np.integer)


def test_spin_multiplicities_is_integer_dtype(charged_jsonl):
    ds = DataSet(charged_jsonl, "energy")
    assert np.issubdtype(ds.spin_multiplicities.dtype, np.integer)


def test_charges_stay_aligned_through_internal_shuffle(charged_jsonl):
    # DataSet shuffles the dataframe before building molecules; labels carry
    # the same value as charge, so equality proves the pairing survived
    ds = DataSet(charged_jsonl, "energy")
    assert list(ds.total_charges) == [int(v) for v in ds.labels]


def test_charges_stay_aligned_through_reservoir_sampling(tmp_path):
    records = [
        {"xyz": XYZ_H2O, "charge": i % 3 - 1, "energy": i} for i in range(N_TOTAL)
    ]
    path = _write_jsonl(tmp_path / "many.jsonl", records)
    ds = DataSet(path, "energy", limit=10)
    assert len(ds) == 10
    assert list(ds.total_charges) == [int(v) % 3 - 1 for v in ds.labels]


def test_integer_valued_floats_are_coerced(tmp_path):
    # JSON numbers arrive as float64; charges must still land as python ints
    path = _write_jsonl(
        tmp_path / "floats.jsonl",
        [
            {"xyz": XYZ_H2O, "charge": -1.0, "energy": 1.0},
            {"xyz": XYZ_CO2, "charge": 0.0, "energy": 2.0},
        ],
    )
    ds = DataSet(path, "energy")
    assert sorted(ds.total_charges) == [-1, 0]
    assert all(isinstance(mol.info["charge"], int) for mol in ds.molecules)


def test_missing_charge_column_defaults_to_neutral(jsonl_file):
    ds = DataSet(jsonl_file, "energy")
    assert list(ds.total_charges) == [0] * N_TOTAL


def test_missing_spin_column_defaults_to_singlet(jsonl_file):
    ds = DataSet(jsonl_file, "energy")
    assert list(ds.spin_multiplicities) == [1] * N_TOTAL


def test_charge_column_without_spin_column(tmp_path):
    path = _write_jsonl(
        tmp_path / "chargeonly.jsonl", [{"xyz": XYZ_H2O, "charge": 1, "energy": 1.0}]
    )
    ds = DataSet(path, "energy")
    assert list(ds.total_charges) == [1]
    assert list(ds.spin_multiplicities) == [1]


def _capture_warnings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nablachem.krr.dataset.warning", lambda msg, **kw: calls.append(kw)
    )
    return calls


def test_missing_charge_column_warns_once(jsonl_file, monkeypatch):
    calls = _capture_warnings(monkeypatch)
    ds = DataSet(jsonl_file, "energy")
    calls.clear()
    ds.total_charges
    ds.total_charges
    ds.total_charges
    assert len(calls) == 1
    assert calls[0]["column"] == "charge"


def test_missing_spin_column_warns_once(jsonl_file, monkeypatch):
    calls = _capture_warnings(monkeypatch)
    ds = DataSet(jsonl_file, "energy")
    calls.clear()
    ds.spin_multiplicities
    ds.spin_multiplicities
    assert len(calls) == 1
    assert calls[0]["column"] == "spin_multiplicity"


def test_present_columns_do_not_warn(charged_jsonl, monkeypatch):
    calls = _capture_warnings(monkeypatch)
    ds = DataSet(charged_jsonl, "energy")
    calls.clear()
    ds.total_charges
    ds.spin_multiplicities
    assert calls == []
