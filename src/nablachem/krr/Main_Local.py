# %% library imports
import time
import jax
import optax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

jax.config.update("jax_enable_x64", True)

# %%


def make_local_data_controller(
    path, prop, holdout_size, chunk_size, limit=12000, rep="cMBDFLocal", label_scale=1.0
):
    import numpy as np
    from nablachem.krr import krr
    from nablachem.krr import dataset
    from nablachem.krr import features
    from nablachem.krr.matrix import ElementalKernelMatrix
    from nablachem.krr.kernels import Gaussian
    from scipy import linalg

    class LocalMatrixOnly(krr.AutoKRR):
        def __init__(
            self,
            X_train,
            X_holdout,
            weights,
            Z_train,
            Z_holdout,
            withtest=False,
            approx=True,
        ):
            self._X_train = X_train
            self._X_holdout = X_holdout
            self._weights = weights
            self._withtest = withtest

            self._train_counts = np.array([rep.shape[0] for rep in X_train])
            self._X_train = np.concatenate(X_train, axis=0)
            self._X_train *= self._weights

            self._holdout_counts = np.array([rep.shape[0] for rep in X_holdout])
            self._X_holdout = np.concatenate(X_holdout, axis=0)
            self._X_holdout *= self._weights

            train_nuclear_charges = np.concatenate(Z_train)
            holdout_nuclear_charges = np.concatenate(Z_holdout)

            if not self._withtest:
                self._X_holdout = None
                holdout_nuclear_charges = None

            self._kernel_matrix = ElementalKernelMatrix(
                self._X_train,
                self._train_counts,
                Gaussian(),
                self._X_holdout,
                self._holdout_counts,
                nuclear_charges=train_nuclear_charges,
                holdout_nuclear_charges=holdout_nuclear_charges,
                approx=approx,
            )

        def ktrain(self, sigma):
            return self._kernel_matrix.compute_train_kernel_matrix(
                sigma, len(self._train_counts)
            )

        def ktest(self, sigma):
            if self._withtest:
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

    # ----- build once -----
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
    rep = local_reps[rep]()
    rep.build([ds])

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
    if chunk_size > n_pool:
        raise ValueError(
            f"chunk_size={chunk_size} cannot exceed training pool size={n_pool}"
        )

    X_holdout = X[holdout_idx]
    Z_holdout = Z[holdout_idx]
    y_test_raw = y[holdout_idx].copy()
    A_holdout = element_counts[holdout_idx]

    # ----- chunk state -----
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
    cache_key = None  # (train_idx tuple, withtest, weights bytes)
    labels_cache = None
    labels_cache_key = None  # train_idx tuple

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

    def get_mlo(weights, withtest, approx=True):
        train_idx = get_chunk()
        w = np.asarray(weights)
        return LocalMatrixOnly(
            X[train_idx], X_holdout, w, Z[train_idx], Z_holdout, withtest, approx=approx
        )

    def get_current_data(weights, withtest):
        nonlocal current_cache, cache_key
        train_idx = get_chunk()

        w = np.asarray(weights)
        key = (tuple(train_idx.tolist()), bool(withtest), w.tobytes())

        if current_cache is not None and cache_key == key:
            _, y_train, y_test, mlo = current_cache
            return y_train, y_test, mlo

        y_train, y_test = get_labels()
        mlo = get_mlo(w, withtest)

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
    }


def estimate_local_model_error(y_train, y_test, mlo, sigma=None, seed=0, xTB=False):

    if xTB == False:
        sigmas_grid = 1.5 ** np.arange(-10, 20)
    else:
        sigmas_grid = [
            2,
        ]
        test_rmse = np.nan
        test_mae = np.nan

    lam = 1e-12
    rng = np.random.default_rng(seed)

    if sigma is not None:
        best_sigma = float(sigma)
        best_rmse = np.nan
    else:
        # 1-split validation inside the chunk
        n = len(y_train)
        perm = rng.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        best_sigma = None
        best_rmse = np.inf

        for s in sigmas_grid:
            K_full = mlo.ktrain(sigma=float(s))
            K_tr_tr = K_full[np.ix_(tr_idx, tr_idx)]
            K_val_tr = K_full[np.ix_(val_idx, tr_idx)]

            alpha = np.linalg.solve(K_tr_tr + lam * np.eye(len(tr_idx)), y_tr)
            pred_val = K_val_tr @ alpha
            rmse_val = np.sqrt(np.mean((pred_val - y_val) ** 2))

            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_sigma = float(s)

    # retrain on full chunk, evaluate on fixed holdout
    K_train = mlo.ktrain(sigma=best_sigma)
    if xTB == False:
        K_test = mlo.ktest(sigma=best_sigma)

        alpha = np.linalg.solve(K_train + lam * np.eye(K_train.shape[0]), y_train)
        pred = K_test @ alpha

        test_rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        test_mae = np.mean(np.abs(pred - y_test))

    return {
        "best_sigma": best_sigma,
        "val_rmse": best_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }


# %%


def _pick_best_workers_for(ctrl, weights, candidates=(4, 6, 8)):
    delta = 2
    y_train, y_test = ctrl["get_labels"]()
    mlo = ctrl["get_mlo"](weights, False, approx=False)
    value = estimate_local_model_error(y_train, y_test, mlo, xTB=True)["val_rmse"]

    def _probe(i):
        weights_delta = weights.at[i].multiply(delta)
        mlo_delta = ctrl["get_mlo"](weights_delta, False, approx=False)
        E = estimate_local_model_error(y_train, y_test, mlo_delta, xTB=True)
        return i, (E["val_rmse"] - value) / delta

    nonzero = [i for i in range(len(weights)) if weights[i] != 0]
    probe = nonzero[: max(4, len(nonzero) // 5)]
    return _pick_best_workers(_probe, probe, candidates)


def _pick_best_workers(fn, probe_indices, candidates=(4, 6, 8)):
    best, best_t = candidates[0], float("inf")
    for n in candidates:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(fn, probe_indices))
        t = time.perf_counter() - t0
        print(f"workers={n}: {t:.2f}s")
        if t < best_t:
            best, best_t = n, t
    return best


def _value_and_grad(ctrl, weights, n_workers=4):
    delta = 2

    y_train, y_test = ctrl["get_labels"]()

    mlo = ctrl["get_mlo"](weights, False, approx=False)
    E_high = estimate_local_model_error(y_train, y_test, mlo, xTB=True)
    value = E_high["val_rmse"]

    grad = np.zeros_like(weights)

    def _compute_grad_i(i):
        if weights[i] == 0:
            return i, 0.0
        weights_delta = weights.at[i].multiply(delta)
        mlo_delta = ctrl["get_mlo"](weights_delta, False, approx=False)
        E = estimate_local_model_error(y_train, y_test, mlo_delta, xTB=True)
        return i, (E["val_rmse"] - value) / delta

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, g in ex.map(_compute_grad_i, range(len(weights))):
            print(i, g)
            grad[i] = g

    return value, grad


class Selector:
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        steps: int,
    ):

        self.lr = learning_rate
        self.batch_size = batch_size
        self.steps = steps
        self.value_log = []
        self.rmse_steps = []
        self.weight_log = []
        self.better_val_errors = []
        self.better_test_errors = []

    def compress(self):

        params = jnp.ones(40)
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        # ---------------------------------------#
        path = "/Users/ali/xTB_data/QM9_with_xtb.jsonl.gz"
        ctrl = make_local_data_controller(
            path,
            "Etot",
            holdout_size=100,
            chunk_size=512,
            limit=512 + 100,
            rep="cMBDFLocal",
            label_scale=627.509474,
        )
        ctrl["next_chunk"]()
        # ---------------------------------------#
        ctrl_xTB = make_local_data_controller(
            path,
            "xtb_E_total",
            holdout_size=100,
            chunk_size=512,
            limit=(512 * self.steps),
            rep="cMBDFLocal",
            label_scale=627.509474,
        )
        # ---------------------------------------#

        ctrl_xTB["next_chunk"]()
        n_workers = _pick_best_workers_for(ctrl_xTB, jnp.array(params))
        ctrl_xTB["next_chunk"]()

        for step in range(self.steps):
            print(f"Step {step}")

            if step % 1 == 0:

                y_train, y_test, mlo = ctrl["get_current_data"](params, True)
                E_high = estimate_local_model_error(
                    y_train, y_test, mlo, sigma=None, xTB=False
                )

                self.rmse_steps.append(step)
                self.better_val_errors.append(E_high["val_rmse"])
                self.better_test_errors.append(E_high["test_rmse"])
                print(f"test_rmse: {E_high['test_rmse']}")
                print(f"val error: {E_high['val_rmse']}")
                print("-----")

            ctrl_xTB["next_chunk"]()
            v, g = _value_and_grad(ctrl_xTB, jnp.array(params), n_workers=n_workers)

            print(g.shape, params.shape)
            updates, opt_state = optimizer.update(jnp.array(g), opt_state, params)
            params = optax.apply_updates(params, updates)

            if sum(params < 0.01) > 5:
                params = params.at[params < 0.01].set(0)

            self.weight_log.append(params)
            self.value_log.append(v)

        return (self.better_test_errors, self.better_val_errors, self.weight_log)

    def plot_logs(self):
        f, axs = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
        axs[0].plot(self.weight_log)
        axs[0].set_title("Weight Log")
        axs[1].semilogy(
            np.cumsum(self.value_log) / (np.arange(len(self.value_log)) + 1)
        )
        axs[1].set_title("xTB val error Log")
        axs[2].set_title("Number of dimensions")
        axs[2].semilogy([(x > 0.1).sum() for x in self.weight_log])
        plt.xlabel("Steps")


s = Selector(
    batch_size=512,
    learning_rate=0.05,
    steps=21,
)

test_errors, val_error, weight_log = s.compress()
s.plot_logs()

test = np.array(test_errors)
val = np.array(val_error)

plt.figure(figsize=(10, 8))
plt.plot(
    np.arange(0, len(test)),
    test,
    marker="o",
    label="Test RMSE",
)

plt.plot(
    np.arange(0, len(val)),
    val,
    marker="s",
    label="Validation RMSE",
)


plt.axhline(
    test_errors[0],
    color="red",
    linestyle="--",
    label="Full Feature Set RMSE",
)
plt.xlabel("steps")
plt.ylabel("B3LYP RMSE")
plt.title("B3LYP RMSE vs steps")
plt.grid(True)
plt.legend()
plt.show()


# %%
