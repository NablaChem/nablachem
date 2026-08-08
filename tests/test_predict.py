import gzip
import json
import math

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet

XYZ_H2O = "3\nH2O\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469"
XYZ_CO2 = "3\nCO2\nC 0.000 0.000 0.000\nO 0.000 0.000 1.160\nO 0.000 0.000 -1.160"
XYZ_CH4 = (
    "5\nCH4\nC 0.000 0.000 0.000\nH 0.000 1.089 0.000\nH 1.026 -0.363 0.000\n"
    "H -0.513 -0.363 0.890\nH -0.513 -0.363 -0.890"
)


def _write_jsonl(path, records, gzipped=False):
    opener = gzip.open if gzipped else open
    with opener(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def main_records():
    return [
        {"xyz": XYZ_H2O, "energy": 1.0},
        {"xyz": XYZ_CO2, "energy": 2.0},
        {"xyz": XYZ_H2O, "energy": 3.0},
        {"xyz": XYZ_CO2, "energy": 4.0},
    ]


@pytest.fixture
def main_file(tmp_path, main_records):
    return _write_jsonl(tmp_path / "main.jsonl", main_records)


# --- prediction-set appending ---


def test_no_predict_defaults(main_file):
    ds = DataSet(main_file, "energy")
    assert ds.n_predict == 0
    assert ds.predict_records is None


def test_predict_appended_at_tail(tmp_path, main_file):
    predict_records = [
        {"xyz": XYZ_CH4, "id": "a"},
        {"xyz": XYZ_H2O, "id": "b"},
    ]
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", predict_records)

    ds = DataSet(main_file, "energy", predict_path=predict_file)

    assert ds.n_predict == 2
    assert len(ds.molecules) == 6  # 4 main + 2 predict
    # predict molecules are the last n_predict, in original order
    assert list(ds.molecules[-2].get_chemical_symbols()) == ["C", "H", "H", "H", "H"]
    assert list(ds.molecules[-1].get_chemical_symbols()) == ["O", "H", "H"]
    # their labels are NaN
    assert np.all(np.isnan(ds.labels[-2:]))
    # original records preserved verbatim, in order
    assert ds.predict_records == predict_records


def test_predict_records_preserve_all_columns_and_order(tmp_path, main_file):
    predict_records = [
        {"xyz": XYZ_H2O, "extra": 42, "note": "first"},
        {"xyz": XYZ_CO2, "extra": 7, "note": "second"},
    ]
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", predict_records)

    ds = DataSet(main_file, "energy", predict_path=predict_file)
    assert ds.predict_records == predict_records


def test_predict_gzip(tmp_path, main_file):
    predict_records = [{"xyz": XYZ_H2O}]
    predict_file = _write_jsonl(
        tmp_path / "predict.jsonl.gz", predict_records, gzipped=True
    )
    ds = DataSet(main_file, "energy", predict_path=predict_file)
    assert ds.n_predict == 1


def test_predict_missing_xyz_errors(tmp_path, main_file):
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", [{"foo": 1}])
    with pytest.raises(SystemExit):
        DataSet(main_file, "energy", predict_path=predict_file)


def test_predict_empty_file(tmp_path, main_file):
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", [])
    ds = DataSet(main_file, "energy", predict_path=predict_file)
    assert ds.n_predict == 0


# --- charge and spin on prediction molecules ---


@pytest.fixture
def charged_main_file(tmp_path):
    records = [
        {"xyz": XYZ_H2O, "energy": 1.0, "charge": -1, "spin_multiplicity": 1},
        {"xyz": XYZ_CO2, "energy": 2.0, "charge": 0, "spin_multiplicity": 3},
    ]
    return _write_jsonl(tmp_path / "charged_main.jsonl", records)


def test_predict_charges_and_spins_are_attached(tmp_path, charged_main_file):
    predict_records = [
        {"xyz": XYZ_CH4, "charge": 2, "spin_multiplicity": 1},
        {"xyz": XYZ_H2O, "charge": -2, "spin_multiplicity": 4},
    ]
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", predict_records)

    ds = DataSet(charged_main_file, "energy", predict_path=predict_file)

    # the last two entries are the prediction molecules
    assert list(ds.total_charges[-2:]) == [2, -2]
    assert list(ds.spin_multiplicities[-2:]) == [1, 4]


def test_predict_without_charge_column_uses_defaults(tmp_path, charged_main_file):
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", [{"xyz": XYZ_CH4}])

    ds = DataSet(charged_main_file, "energy", predict_path=predict_file)

    assert ds.total_charges[-1] == 0  # neutral
    assert ds.spin_multiplicities[-1] == 1  # singlet


# --- write_predictions_jsonl ---


def test_write_predictions_roundtrip(tmp_path, main_file):
    predict_records = [
        {"xyz": XYZ_H2O, "id": "a"},
        {"xyz": XYZ_CO2, "id": "b"},
    ]
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", predict_records)
    ds = DataSet(main_file, "energy", predict_path=predict_file)

    ds.write_predictions_jsonl(np.array([1.5, 2.5]), "property", predict_file)

    with open(predict_file) as f:
        out = [json.loads(line) for line in f]
    assert len(out) == 2
    assert out[0] == {"xyz": XYZ_H2O, "id": "a", "property": 1.5}
    assert out[1] == {"xyz": XYZ_CO2, "id": "b", "property": 2.5}


def test_write_predictions_nan_becomes_null(tmp_path, main_file):
    predict_records = [{"xyz": XYZ_H2O}]
    predict_file = _write_jsonl(tmp_path / "predict.jsonl", predict_records)
    ds = DataSet(main_file, "energy", predict_path=predict_file)

    ds.write_predictions_jsonl(np.array([math.nan]), "property", predict_file)

    with open(predict_file) as f:
        out = json.loads(f.readline())
    assert out["property"] is None


def test_write_predictions_preserves_gzip(tmp_path, main_file):
    predict_records = [{"xyz": XYZ_H2O}]
    predict_file = _write_jsonl(
        tmp_path / "predict.jsonl.gz", predict_records, gzipped=True
    )
    ds = DataSet(main_file, "energy", predict_path=predict_file)

    ds.write_predictions_jsonl(np.array([3.14]), "property", predict_file)

    with gzip.open(predict_file, "rt") as f:
        out = json.loads(f.readline())
    assert out["property"] == 3.14


# --- end-to-end AutoKRR behaviour with an appended prediction tail ---
#
# Mirrors the mock harness from tests/test_matrix.py: a 1-D "representation" so
# AutoKRR uses the global kernel path, forced hyperparameters, detrending off.

from nablachem.krr import kernels  # noqa: E402
from nablachem.krr.krr import AutoKRR  # noqa: E402
from nablachem.krr.matrix import GlobalKernelMatrix  # noqa: E402

N_TRAIN_MAX = 8
N_REAL_HOLDOUT = 4
SIGMA_MAP = {4: 0.5, 8: 3.0}
LAM = 1e-6


class _MockRepresenter:
    def __init__(self, X):
        self._molecules = list(range(len(X)))
        self._X = X

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.compute(self._molecules[key])
        return [self._X[key]]

    def compute(self, mol_indices):
        return [self._X[i] for i in mol_indices]


class _MockDataset:
    def __init__(self, X, y, n_predict=0):
        self.representations = _MockRepresenter(X)
        self.labels = y.copy()
        self.n_predict = n_predict
        self._n = len(X)

    def __len__(self):
        return self._n


class _ForcedSigmaKRR(AutoKRR):
    def _optimize_hyperparameters(self, ntrain, _length_heuristic):
        return {"sigma": SIGMA_MAP[ntrain], "lambda": LAM}, 1.0, 1.0


def _run(X, y, n_predict):
    return _ForcedSigmaKRR(
        _MockDataset(X, y, n_predict=n_predict),
        mincount=4,
        maxcount=8,
        kernel_func=kernels.Gaussian(),
        detrending=(),
    )


def test_appended_predictions_do_not_change_holdout_metrics():
    n_all = N_TRAIN_MAX + N_REAL_HOLDOUT
    X_base = np.arange(n_all, dtype=float).reshape(-1, 1)
    y_base = X_base[:, 0].copy()

    baseline = _run(X_base, y_base, n_predict=0)

    # Append two prediction rows (duplicates of holdout rows 0 and 1) with NaN labels.
    holdout_start = N_TRAIN_MAX
    X_pred = np.vstack([X_base, X_base[holdout_start], X_base[holdout_start + 1]])
    y_pred = np.concatenate([y_base, [np.nan, np.nan]])
    with_predict = _run(X_pred, y_pred, n_predict=2)

    # Real-holdout residuals and metrics must be identical.
    assert np.allclose(
        baseline.holdout_residuals[8], with_predict.holdout_residuals[8]
    )
    assert len(with_predict.holdout_residuals[8]) == N_REAL_HOLDOUT
    assert baseline.results[8]["test_rmse"] == pytest.approx(
        with_predict.results[8]["test_rmse"]
    )
    assert baseline.results[1]["test_rmse"] == pytest.approx(
        with_predict.results[1]["test_rmse"]
    )


def test_training_size_cannot_exceed_labeled_molecules():
    # 6 labeled + 10 predict rows; maxcount forces max_training_size=8 > 6 labeled.
    # Must error instead of pulling NaN-labeled predict rows into training.
    n_labeled, n_pred = 6, 10
    X = np.arange(n_labeled + n_pred, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()
    y[n_labeled:] = np.nan
    with pytest.raises(SystemExit):
        _ForcedSigmaKRR(
            _MockDataset(X, y, n_predict=n_pred),
            mincount=8,
            maxcount=8,
            kernel_func=kernels.Gaussian(),
            detrending=(),
        )


def test_evaluate_models_real_holdout_and_predict_tail_manual():
    # 8 train + 4 real holdout, plus 3 DISTINCT predict rows with NaN labels.
    # Reconstruct the reference model by hand and check the split explicitly.
    n_all = N_TRAIN_MAX + N_REAL_HOLDOUT
    X_base = np.arange(n_all, dtype=float).reshape(-1, 1)
    y_base = X_base[:, 0].copy()
    X_predict = np.array([[2.5], [5.5], [20.0]])
    n_pred = len(X_predict)

    X = np.vstack([X_base, X_predict])
    y = np.concatenate([y_base, np.full(n_pred, np.nan)])
    krr = _run(X, y, n_predict=n_pred)

    ntrain = N_TRAIN_MAX
    sigma = SIGMA_MAP[ntrain]

    # --- manual reference model (detrend off, global kernel) ---
    kmat = GlobalKernelMatrix(X_base[:ntrain], kernels.Gaussian())
    shift = np.mean(y_base[:ntrain])
    y_train_c = y_base[:ntrain] - shift

    K_train = kmat.compute_train_kernel_matrix(sigma, ntrain)
    col_mean = K_train.mean(axis=0, keepdims=True)
    overall = K_train.mean()
    K_train_c = K_train - K_train.mean(axis=1, keepdims=True) - col_mean + overall
    alpha = np.linalg.solve(K_train_c + LAM * np.eye(ntrain), y_train_c)

    def abs_pred(X_test):
        K_test = kmat.compute_test_kernel_matrix(sigma, ntrain, X_test)
        K_test_c = K_test - K_test.mean(axis=1, keepdims=True) - col_mean + overall
        return K_test_c @ alpha + shift

    expected_real_abs = abs_pred(X_base[ntrain:])
    expected_pred_abs = abs_pred(X_predict)
    expected_real_resid = y_base[ntrain:] - expected_real_abs

    # (1) residuals + metrics cover ONLY the real holdout rows
    assert len(krr.holdout_residuals[ntrain]) == N_REAL_HOLDOUT
    assert np.allclose(krr.holdout_residuals[ntrain], expected_real_resid)
    assert krr.results[ntrain]["test_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(expected_real_resid ** 2)))
    )
    assert krr.results[ntrain]["test_mae"] == pytest.approx(
        float(np.mean(np.abs(expected_real_resid)))
    )
    # nullmodel test metrics also restricted to the real holdout
    mean_pred = np.mean(y_base[:ntrain])
    assert krr.results[1]["test_rmse"] == pytest.approx(
        float(np.sqrt(np.mean((mean_pred - y_base[ntrain:]) ** 2)))
    )

    # (2) prediction tail: absolute-valued predictions, length n_predict
    hp = krr.holdout_predictions[ntrain]
    assert len(hp) == N_REAL_HOLDOUT + n_pred
    predict_tail = hp[N_REAL_HOLDOUT:]
    assert len(predict_tail) == n_pred
    assert np.all(np.isfinite(predict_tail))
    assert np.allclose(predict_tail, expected_pred_abs)
    # the real-holdout slice of holdout_predictions is absolute (true - residual),
    # i.e. distinct from the stored residuals
    assert np.allclose(hp[:N_REAL_HOLDOUT], expected_real_abs)
    assert not np.allclose(hp[:N_REAL_HOLDOUT], krr.holdout_residuals[ntrain])


def test_prediction_of_duplicate_holdout_row_matches():
    n_all = N_TRAIN_MAX + N_REAL_HOLDOUT
    X_base = np.arange(n_all, dtype=float).reshape(-1, 1)
    y_base = X_base[:, 0].copy()
    holdout_start = N_TRAIN_MAX

    X_pred = np.vstack([X_base, X_base[holdout_start], X_base[holdout_start + 1]])
    y_pred = np.concatenate([y_base, [np.nan, np.nan]])
    krr = _run(X_pred, y_pred, n_predict=2)

    preds = krr.holdout_predictions[8]
    assert len(preds) == N_REAL_HOLDOUT + 2

    # A prediction row identical to a real-holdout row must yield the identical
    # reconstructed (original-unit) prediction.
    assert preds[N_REAL_HOLDOUT + 0] == pytest.approx(preds[0])
    assert preds[N_REAL_HOLDOUT + 1] == pytest.approx(preds[1])
    # Reconstructed predictions are finite for the prediction tail.
    assert np.all(np.isfinite(preds[N_REAL_HOLDOUT:]))
