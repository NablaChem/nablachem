#!/usr/bin/env python3
import warnings
import jax
import optax
import numpy as np
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy import linalg as _scipy_linalg
import psutil, os

jax.config.update("jax_enable_x64", True)


def _ram(label=""):
    proc = psutil.Process(os.getpid())
    used = proc.memory_info().rss / 1e9
    total = psutil.virtual_memory().total / 1e9
    print(f"[RAM] {label}: {used:.2f} GB used / {total:.1f} GB total", flush=True)


def make_local_data_controller(
    path,
    prop,
    holdout_size,
    chunk_size,
    limit=12000,
    rep="cMBDFLocal",
    label_scale=1.0,
    kernel="elemental",
    mace_model_path="medium",
):
    import numpy as np
    from nablachem.krr import krr
    from nablachem.krr import dataset
    from nablachem.krr import features
    from nablachem.krr.matrix import ElementalKernelMatrix, LocalKernelMatrix
    from nablachem.krr.kernels import Gaussian
    from scipy import linalg

    _kernel_classes = {
        "elemental": ElementalKernelMatrix,
        "local": LocalKernelMatrix,
    }
    if kernel not in _kernel_classes:
        raise ValueError(f"kernel must be one of {list(_kernel_classes.keys())}")
    KernelMatrixClass = _kernel_classes[kernel]

    class LocalMatrixOnly(krr.AutoKRR):
        def __init__(
            self,
            X_train,
            X_holdout,
            weights,
            Z_train,
            Z_holdout,
            approx=True,
        ):
            self._X_train = X_train
            self._X_holdout = X_holdout
            self._weights = weights

            mask = weights != 0

            self._train_counts = np.array([rep.shape[0] for rep in X_train])
            self._X_train = np.concatenate(X_train, axis=0)
            self._X_train = self._X_train[:, mask] * weights[mask]

            self._holdout_counts = np.array([rep.shape[0] for rep in X_holdout])
            self._X_holdout = np.concatenate(X_holdout, axis=0)
            self._X_holdout = self._X_holdout[:, mask] * weights[mask]

            train_nuclear_charges = np.concatenate(Z_train)
            holdout_nuclear_charges = np.concatenate(Z_holdout)

            kwargs = dict(approx=approx)
            if kernel == "elemental":
                kwargs["nuclear_charges"] = train_nuclear_charges
                kwargs["holdout_nuclear_charges"] = holdout_nuclear_charges

            self._kernel_matrix = KernelMatrixClass(
                self._X_train,
                self._train_counts,
                Gaussian(),
                self._X_holdout,
                self._holdout_counts,
                **kwargs,
            )

        def ktrain(self, sigma):
            return self._kernel_matrix.compute_train_kernel_matrix(
                sigma, len(self._train_counts)
            )

        def ktest(self, sigma):
            batches = []
            batch = 0
            while True:
                K = self._kernel_matrix.compute_test_kernel_matrix(
                    sigma, len(self._train_counts), batch
                )
                if K is None:
                    break
                batches.append(K)
                batch += 1
            return np.vstack(batches)

    ds = dataset.DataSet(path, prop, limit=limit, select=None)
    local_reps = {
        "MBDFLocal": features.MBDFLocal,
        "cMBDFLocal": features.cMBDFLocal,
        "SLATMLocal": features.SLATMLocal,
        "MACELocal": features.MACELocal,
        "FCHL19Local": features.FCHL19Local,
    }
    if rep not in local_reps:
        raise ValueError(f"rep must be one of {list(local_reps.keys())}")
    rep_kwargs = {"model_path": mace_model_path} if rep.startswith("MACE") else {}
    rep = local_reps[rep](**rep_kwargs)
    _ram(f"before rep.build limit={limit}")
    rep.build([ds])
    _ram(f"after rep.build limit={limit}")

    X = np.array(ds.representations, dtype="object")
    Z = np.array(ds.nuclear_charges, dtype="object")
    y = ds.labels * label_scale
    element_counts, _ = ds.get_element_counts()

    n_total = len(y)
    if holdout_size >= n_total:
        raise ValueError(f"holdout_size={holdout_size} must be < total={n_total}")

    holdout_start = n_total - holdout_size
    holdout_idx = np.arange(holdout_start, n_total)
    train_pool_idx = np.arange(0, holdout_start)
    n_pool = len(train_pool_idx)

    if n_pool == 0:
        raise ValueError("No training data left after reserving holdout.")
    if chunk_size >= n_pool:
        raise ValueError(
            f"chunk_size={chunk_size} must be less than training pool size={n_pool}"
        )

    X_holdout = X[holdout_idx]
    Z_holdout = Z[holdout_idx]
    y_test_raw = y[holdout_idx].copy()
    A_holdout = element_counts[holdout_idx]

    pos = 0
    current_train_idx = None

    def _compute_next_indices():
        nonlocal pos
        if pos + chunk_size <= n_pool:
            idx = train_pool_idx[pos : pos + chunk_size]
            pos += chunk_size
            if pos == n_pool:
                pos = 0
        else:
            r = (pos + chunk_size) - n_pool
            idx = np.concatenate([train_pool_idx[pos:], train_pool_idx[:r]])
            pos = r
        return idx

    def next_chunk():
        nonlocal current_train_idx, current_cache, labels_cache, labels_cache_key
        current_train_idx = _compute_next_indices()
        current_cache = None
        labels_cache = None
        labels_cache_key = None
        return current_train_idx

    def set_chunk(train_idx):
        nonlocal current_train_idx, current_cache, labels_cache, labels_cache_key
        train_idx = np.asarray(train_idx)
        if np.any(train_idx >= holdout_start):
            raise ValueError("train_idx includes holdout indices.")
        current_train_idx = train_idx
        current_cache = None
        labels_cache = None
        labels_cache_key = None
        return current_train_idx

    def get_chunk():
        nonlocal current_train_idx
        if current_train_idx is None:
            next_chunk()
        return current_train_idx

    current_cache = None
    cache_key = None
    labels_cache = None
    labels_cache_key = None

    def get_labels():
        nonlocal labels_cache, labels_cache_key
        train_idx = get_chunk()
        key = tuple(train_idx.tolist())

        if labels_cache is not None and labels_cache_key == key:
            return labels_cache

        y_train_raw = y[train_idx].copy()
        A_train = element_counts[train_idx]
        coefs = linalg.lstsq(A_train, y_train_raw)[0]
        y_train = y_train_raw - (A_train @ coefs)
        y_test = y_test_raw - (A_holdout @ coefs)

        labels_cache = (y_train, y_test)
        labels_cache_key = key
        return labels_cache

    def get_mlo(weights, approx=True):
        train_idx = get_chunk()
        w = np.asarray(weights)
        return LocalMatrixOnly(
            X[train_idx], X_holdout, w, Z[train_idx], Z_holdout, approx=approx
        )

    def get_current_data(weights):
        nonlocal current_cache, cache_key
        train_idx = get_chunk()

        w = np.asarray(weights)
        key = (tuple(train_idx.tolist()), w.tobytes())

        if current_cache is not None and cache_key == key:
            _, y_train, y_test, mlo = current_cache
            return y_train, y_test, mlo

        y_train, y_test = get_labels()
        mlo = get_mlo(w)

        current_cache = (train_idx, y_train, y_test, mlo)
        cache_key = key
        return y_train, y_test, mlo

    return {
        "next_chunk": next_chunk,
        "set_chunk": set_chunk,
        "get_chunk": get_chunk,
        "get_labels": get_labels,
        "get_mlo": get_mlo,
        "get_current_data": get_current_data,
        "n_features": X[0].shape[-1],
    }


def _grid_search_hyperparams(mlo, y_train, n_workers=11):
    """Grid search for best (sigma, lambda) via shuffled cross-validation.

    Returns (best_sigma, best_lam, best_val_rmse). Returns (None, None, nan)
    if no valid hyperparameter combination is found.
    """
    n = len(y_train)
    factors, lam_grid = mlo.get_hyperparameter_grid(n)
    length_heuristic = mlo._kernel_matrix.length_scale(n)
    validation = mlo.validation_size(n)
    shufs = 50

    y = y_train - np.mean(y_train)

    tqdm.write(
        f"[hyperparam] grid search: {len(factors)} sigmas x {len(lam_grid)} lambdas"
        f"  n={n}  workers={n_workers}",
        end="\n",
    )

    def _eval_factor(factor):
        sigma = length_heuristic * factor
        K = mlo.ktrain(sigma)
        # kernel centering
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        K_mean = K.mean()
        K = K - row_mean - col_mean + K_mean

        eigvals, Q = np.linalg.eigh(K)
        cond = eigvals[-1] / eigvals[1] if eigvals[1] > 0 else np.inf
        if cond > 1e15:
            tqdm.write(
                f"  [sigma={sigma:.3g}] skipped (ill-conditioned, cond={cond:.1e})"
            )
            return []

        use_schur = cond < 5e8 and n > 128

        factor_results = []
        idx = np.arange(n)  # thread-local copy — no shared mutable state
        rng = np.random.default_rng()  # thread-local RNG

        for lam in lam_grid:
            if use_schur:
                Kinv = (Q * (1.0 / (eigvals + lam))) @ Q.T

            split_rmse = []
            for _ in range(shufs):
                rng.shuffle(idx)
                y_s = y[idx]

                if use_schur:
                    Ks = Kinv[np.ix_(idx, idx)]
                    E = Ks[:-validation, :-validation]
                    H = Ks[-validation:, -validation:]
                    F = Ks[:-validation, -validation:]
                    G = Ks[-validation:, :-validation]
                    alpha = E @ y_s[:-validation] - F @ (
                        np.linalg.inv(H) @ (G @ y_s[:-validation])
                    )
                else:
                    K_s = K[np.ix_(idx, idx)]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", _scipy_linalg.LinAlgWarning)
                            alpha = _scipy_linalg.solve(
                                K_s[:-validation, :-validation]
                                + lam * np.eye(n - validation),
                                y_s[:-validation],
                                assume_a="pos",
                            )
                    except Exception:
                        continue

                pred = K[np.ix_(idx[-validation:], idx[:-validation])] @ alpha
                split_rmse.append(np.sqrt(((pred - y_s[-validation:]) ** 2).mean()))

                if len(split_rmse) > 5:
                    one = np.median(split_rmse[::2])
                    two = np.median(split_rmse[1::2])
                    if abs(one - two) / np.median(split_rmse) < 5e-2:
                        break

            if len(split_rmse) < 5:
                continue
            avg = np.median(split_rmse)
            factor_results.append((sigma, lam, avg))

        if factor_results:
            best = min(factor_results, key=lambda x: x[2])
            tqdm.write(
                f"  [sigma={sigma:.3g}] done — best lam={best[1]:.1e}  rmse={best[2]:.4f}"
                f"  (schur={use_schur})"
            )
        return factor_results

    best_sigma, best_lam, best_rmse = None, None, np.inf

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        pbar = tqdm(
            ex.map(_eval_factor, factors),
            total=len(factors),
            desc="hyperparam",
        )
        for factor_results in pbar:
            for sigma, lam, avg in factor_results:
                if avg < best_rmse:
                    best_rmse, best_sigma, best_lam = avg, sigma, lam
                    pbar.set_postfix(
                        sigma=f"{best_sigma:.3g}",
                        lam=f"{best_lam:.1e}",
                        rmse=f"{best_rmse:.4f}",
                    )

    tqdm.write(
        f"[hyperparam] best: sigma={best_sigma:.3g}  lam={best_lam:.1e}"
        f"  val_rmse={best_rmse:.4f}"
    )
    return best_sigma, best_lam, best_rmse


def _solve_and_predict(mlo, y_train, y_test, sigma, lam):
    """Fit KRR with given (sigma, lam) and return (test_rmse, test_mae).

    Uses kernel centering consistent with the hyperparameter search.
    Returns (nan, nan) on numerical failure.
    """
    K_train = mlo.ktrain(sigma)
    col_mean = K_train.mean(axis=0, keepdims=True)
    K_mean = K_train.mean()
    K_train_c = K_train - K_train.mean(axis=1, keepdims=True) - col_mean + K_mean

    shift = np.mean(y_train)
    y = y_train - shift
    try:
        alpha = np.linalg.solve(K_train_c + lam * np.eye(len(y)), y)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    K_test = mlo.ktest(sigma)
    K_test_c = K_test - K_test.mean(axis=1, keepdims=True) - col_mean + K_mean
    pred = K_test_c @ alpha + shift

    test_rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    test_mae = np.mean(np.abs(pred - y_test))
    return test_rmse, test_mae


def estimate_local_model_error(
    y_train, y_test, mlo, xTB=False, xtb_cache=None, n_workers=11
):
    """Estimate KRR test error.

    xTB=False: run full hyperparameter grid search then evaluate on y_test.
    xTB=True:  use pre-cached (sigma, lam) from xtb_cache and evaluate on y_test.
    """
    _nan = {
        "best_sigma": np.nan,
        "best_lam": np.nan,
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "test_mae": np.nan,
    }

    if xTB:
        sigma, lam = xtb_cache[0]
        test_rmse, test_mae = _solve_and_predict(mlo, y_train, y_test, sigma, lam)
        return {
            "best_sigma": sigma,
            "best_lam": lam,
            "val_rmse": np.nan,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
    else:
        sigma, lam, val_rmse = _grid_search_hyperparams(
            mlo, y_train, n_workers=n_workers
        )
        if sigma is None:
            return _nan
        test_rmse, test_mae = _solve_and_predict(mlo, y_train, y_test, sigma, lam)
        return {
            "best_sigma": sigma,
            "best_lam": lam,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }


def _value_and_grad(ctrl, weights, n_workers=11, xtb_cache=None):
    y_train, y_test = ctrl["get_labels"]()

    mlo = ctrl["get_mlo"](weights, approx=False)
    E_high = estimate_local_model_error(
        y_train, y_test, mlo, xTB=True, xtb_cache=xtb_cache
    )
    value = E_high["test_rmse"]

    grad = np.zeros_like(weights)

    def _compute_grad_i(i):
        if weights[i] == 0:
            return i, 0.0
        eps = abs(float(weights[i])) * 1e-2
        mlo_fwd = ctrl["get_mlo"](weights.at[i].add(eps), approx=False)
        mlo_bwd = ctrl["get_mlo"](weights.at[i].add(-eps), approx=False)
        E_fwd = estimate_local_model_error(
            y_train, y_test, mlo_fwd, xTB=True, xtb_cache=xtb_cache
        )
        E_bwd = estimate_local_model_error(
            y_train, y_test, mlo_bwd, xTB=True, xtb_cache=xtb_cache
        )
        return i, (E_fwd["test_rmse"] - E_bwd["test_rmse"]) / (2 * eps)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        pbar = tqdm(
            ex.map(_compute_grad_i, range(len(weights))),
            total=len(weights),
            desc="grad",
        )
        for i, g in pbar:
            grad[i] = g
            pbar.set_postfix(i=i, g=f"{g:.4f}")

    return value, grad


class Selector:
    def __init__(
        self,
        learning_rate: float,
        steps: int,
        n_runs: int = 5,
        hq_chunk_size: int = 256,
        lq_chunk_size: int = 256,
        rep: str = "cMBDFLocal",
        kernel: str = "elemental",
        n_workers: int = None,
        mace_model_path: str = "medium",
    ):
        self.lr = learning_rate
        self.steps = steps
        self.n_runs = n_runs
        self.hq_chunk_size = hq_chunk_size
        self.lq_chunk_size = lq_chunk_size
        self.rep = rep
        self.kernel = kernel
        self.n_workers = n_workers
        self.mace_model_path = mace_model_path

    def _run_once(self, n_workers, path, seed=0):
        np.random.seed(seed)

        ctrl = make_local_data_controller(
            path,
            "Etot",
            holdout_size=self.hq_chunk_size,
            chunk_size=self.hq_chunk_size,
            limit=self.hq_chunk_size * 10 + self.hq_chunk_size,
            rep=self.rep,
            label_scale=627.509474,
            kernel=self.kernel,
            mace_model_path=self.mace_model_path,
        )
        ctrl["next_chunk"]()

        params = jnp.ones(ctrl["n_features"])
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        ctrl_xTB = make_local_data_controller(
            path,
            "xtb_E_total",
            holdout_size=self.lq_chunk_size,
            chunk_size=self.lq_chunk_size,
            limit=self.lq_chunk_size * (2 * self.steps + 4),
            rep=self.rep,
            label_scale=627.509474,
            kernel=self.kernel,
            mace_model_path=self.mace_model_path,
        )

        # One-time hyperparameter search on 4× training data before the main loop.
        # Result is cached and reused for all gradient steps; no test eval here.
        xtb_cache = [None]
        ctrl_xTB["set_chunk"](np.arange(self.lq_chunk_size))
        y_init, _ = ctrl_xTB["get_labels"]()
        mlo_init = ctrl_xTB["get_mlo"](jnp.ones(ctrl_xTB["n_features"]), approx=False)
        sigma_init, lam_init, _ = _grid_search_hyperparams(
            mlo_init, y_init, n_workers=n_workers
        )
        xtb_cache[0] = (sigma_init, lam_init)
        ctrl_xTB["next_chunk"]()

        test_errors, val_errors, weight_log, value_log, rmse_steps = [], [], [], [], []

        zero_threshold = 0.0
        plateau_window = 10
        _window_count = 0
        _prev_active_dims = len(params)
        active_dims = len(params)

        pbar = tqdm(range(self.steps), desc="compress", unit="step", dynamic_ncols=True)
        for step in pbar:

            if step % 5 == 0:
                y_train, y_test, mlo = ctrl["get_current_data"](params)
                E_high = estimate_local_model_error(
                    y_train, y_test, mlo, n_workers=n_workers
                )
                active_dims = int((np.array(params) > 0.001).sum())
                rmse_steps.append(step)
                val_errors.append(E_high["val_rmse"])
                test_errors.append(E_high["test_rmse"])
                tqdm.write(
                    f"  step {step:>4d} │ test RMSE {E_high['test_rmse']:.4f}"
                    f"  val RMSE {E_high['val_rmse']:.4f}"
                    f"  test MAE {E_high['test_mae']:.4f}"
                    f"  dims {active_dims}  zero_thr {zero_threshold:.4f}"
                )
                pbar.set_postfix(
                    test=f"{E_high['test_rmse']:.4f}",
                    val=f"{E_high['val_rmse']:.4f}",
                    dims=active_dims,
                )

            ctrl_xTB["next_chunk"]()
            v, g = _value_and_grad(
                ctrl_xTB, jnp.array(params), n_workers=n_workers, xtb_cache=xtb_cache
            )

            _window_count += 1
            if _window_count == plateau_window:
                if (
                    _prev_active_dims is not None
                    and abs(active_dims - _prev_active_dims) <= 3
                ):
                    nonzero = np.array(params)[np.array(params) > 0]
                    if len(nonzero) > 0:
                        zero_threshold = float(np.percentile(nonzero, 0.1))
                        tqdm.write(
                            f"  [plateau] step {step} — active dims change"
                            f" {abs(active_dims - _prev_active_dims)} <= 3"
                            f" → zero_threshold = {zero_threshold:.4f}"
                        )
                _prev_active_dims = active_dims
                _window_count = 0

            updates, opt_state = optimizer.update(jnp.array(g), opt_state, params)
            params = optax.apply_updates(params, updates)

            if zero_threshold > 0:
                params = params.at[params < zero_threshold].set(0)

            weight_log.append(np.array(params))
            value_log.append(v)

        return test_errors, val_errors, weight_log, value_log, rmse_steps

    def compress(self, path, output_path):
        n_workers = self.n_workers
        print(f"Workers        : {n_workers}")

        all_test_errors, all_val_errors, all_weight_logs, all_value_logs = (
            [],
            [],
            [],
            [],
        )
        rmse_steps = None

        seeds = [1, 2, 3, 4, 5]
        for run in range(self.n_runs):
            print(
                f"\n{'='*40}\n  Run {run + 1}/{self.n_runs}  (seed={seeds[run]})\n{'='*40}"
            )
            t_err, v_err, w_log, val_log, steps = self._run_once(
                n_workers, path, seed=seeds[run]
            )
            all_test_errors.append(np.array(t_err))
            all_val_errors.append(np.array(v_err))
            all_weight_logs.append(w_log)
            all_value_logs.append(np.array(val_log))
            if rmse_steps is None:
                rmse_steps = steps

        results = {
            "rmse_steps": np.array(rmse_steps),
            "test_errors": np.array(all_test_errors),
            "val_errors": np.array(all_val_errors),
            "value_log": np.array(all_value_logs),
            "weight_log": np.array(
                [
                    [
                        np.array(all_weight_logs[r][i])
                        for i in range(len(all_weight_logs[0]))
                    ]
                    for r in range(self.n_runs)
                ]
            ),
        }

        np.savez(
            f"{output_path}_{self.rep}_{self.kernel}_{self.hq_chunk_size}", **results
        )
        print(
            f"\nResults saved to {output_path}_{self.rep}_{self.kernel}_{self.hq_chunk_size}.npz"
        )
        return results


if __name__ == "__main__":
    import argparse
    import os
    import psutil

    mem = psutil.virtual_memory()
    print(f"CPUs available : {os.cpu_count()}")
    print(f"RAM total      : {mem.total / 1e9:.1f} GB")
    print(f"RAM available  : {mem.available / 1e9:.1f} GB")
    try:
        import subprocess

        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        print(f"GPU            : {gpu_info}")
    except Exception:
        print("GPU            : none / nvidia-smi not available")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/ali/xTB_data/QM9_with_xtb.jsonl.gz")
    parser.add_argument("--output", default="results")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hq-chunk-size", type=int, default=256)
    parser.add_argument("--lq-chunk-size", type=int, default=256)
    parser.add_argument(
        "--rep",
        default="cMBDFLocal",
        choices=["MBDFLocal", "cMBDFLocal", "SLATMLocal", "MACELocal", "FCHL19Local"],
    )
    parser.add_argument(
        "--kernel",
        default="elemental",
        choices=["elemental", "local"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--mace-model",
        default="medium",
        help="MACE model name or path to a local .model file (required on nodes without internet).",
    )
    args = parser.parse_args()

    s = Selector(
        learning_rate=args.lr,
        steps=args.steps,
        n_runs=args.n_runs,
        hq_chunk_size=args.hq_chunk_size,
        lq_chunk_size=args.lq_chunk_size,
        rep=args.rep,
        kernel=args.kernel,
        n_workers=args.workers,
        mace_model_path=args.mace_model,
    )
    s.compress(path=args.data, output_path=args.output)
