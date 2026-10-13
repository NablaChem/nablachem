import numpy as np
import pytest

from nablachem.krr import kernels
from nablachem.krr.krr import AutoKRR


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
    def __init__(self, X, y):
        self.representations = _MockRepresenter(X)
        self.labels = y.copy()
        self.n_predict = 0
        self._n = len(X)

    def __len__(self):
        return self._n


def _dataset():
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(96, 2))
    # heavy-tailed noise so that RMSE and MAE favour different hyperparameters
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_t(1.5, size=96)
    return _MockDataset(X, y)


def _run(**kwargs):
    return AutoKRR(
        _dataset(),
        mincount=32,
        maxcount=64,
        kernel_func=kernels.Gaussian(),
        detrending=(),
        seed=0,
        **kwargs,
    )


def test_default_is_rmse():
    assert _run().results == _run(cv_metric="rmse").results


@pytest.mark.parametrize("metric", ["rmse", "mae"])
def test_selection_is_argmin_of_metric(metric):
    krr = _run(cv_metric=metric)
    for ntrain in (32, 64):
        entries = [e for e in krr._archive["hyperopt"] if e["ntrain"] == ntrain]
        best = min(entries, key=lambda e: np.median(e[f"val_{metric}"]))
        result = krr.results[ntrain]
        assert result["parameters"] == {
            "sigma": best["sigma"],
            "lambda": best["lambda"],
        }
        assert result["val_rmse"] == pytest.approx(np.median(best["val_rmse"]))
        assert result["val_mae"] == pytest.approx(np.median(best["val_mae"]))


def test_unknown_metric_errors():
    with pytest.raises(SystemExit):
        _run(cv_metric="foo")
