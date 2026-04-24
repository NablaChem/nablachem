import warnings
import numpy as np
import time
import json
from scipy import linalg
from scipy.linalg import LinAlgWarning
from . import utils
from . import matrix
from .dataset import DataSet
from . import kernels


class AutoKRR:
    def __init__(
        self,
        dataset: DataSet,
        mincount: int,
        maxcount: int,
        kernel_func: kernels.Kernel,
        detrend_atomic: bool = True,
        detrend_pairs: str | None = None,
        elemental: bool = False,
    ) -> None:
        self._archive = {}
        self._archive["hyperopt"] = []
        self.dataset = dataset
        self._training_sizes = utils.get_training_sizes(mincount, maxcount)
        self._detrend_atomic = detrend_atomic
        self._detrend_pairs = detrend_pairs
        self._elemental = elemental

        self._create_holdout_split(elemental)

        self.results: dict[int, dict[str, float]] = {}
        self.holdout_residuals: dict[int, np.ndarray] = {}
        self._add_nullmodel()

        if self._local:
            if self._elemental:
                self._kernel_matrix = matrix.ElementalKernelMatrix(
                    self._X_train,
                    self._train_counts,
                    kernel_func,
                    nuclear_charges=self._train_nuclear_charges,
                )
            else:
                self._kernel_matrix = matrix.LocalKernelMatrix(
                    self._X_train,
                    self._train_counts,
                    kernel_func,
                )
        else:
            if self._elemental:
                utils.error("--elemental is not supported for global representations")
            self._kernel_matrix = matrix.GlobalKernelMatrix(self._X_train, kernel_func)

        last_rmse = None
        last_size = None
        best_cases = {}

        for i, ntrain in enumerate(self._training_sizes):
            length_heuristic = self._kernel_matrix.length_scale(ntrain)
            best_parameters, best_val_rmse, best_val_mae, eig_count, direct_count = (
                self._optimize_hyperparameters(ntrain, length_heuristic)
            )
            best_cases[ntrain] = best_parameters

            improvement = {}
            if last_rmse is not None:
                improvement["validation_slope"] = float(
                    np.log(best_val_rmse / last_rmse) / np.log(ntrain / last_size)
                )
            else:
                improvement["validation_slope"] = None

            last_rmse = best_val_rmse
            last_size = ntrain

            utils.info(
                "Training size completed",
                ntrain=ntrain,
                validation_rmse=float(best_val_rmse),
                eig_count=eig_count,
                direct_count=direct_count,
                **improvement,
            )

            self.results[ntrain] = {
                "parameters": best_parameters,
                "val_rmse": float(best_val_rmse),
                "val_mae": float(best_val_mae),
                "eig_count": eig_count,
                "direct_count": direct_count,
                **improvement,
            }

        utils.info("Evaluate models on test set")
        self._evaluate_models(best_cases)

    def store_archive(self, filename: str, metadata: dict) -> None:
        """Store hyperparameter optimization archive and learning curve data to JSON file"""
        # Add learning curve data to archive
        learning_curve_data = []

        # Add nullmodel (ntrain=1)
        if 1 in self.results:
            learning_curve_data.append(
                {
                    "ntrain": 1,
                    "val_rmse": self.results[1]["val_rmse"],
                    "test_rmse": self.results[1]["test_rmse"],
                    "val_mae": self.results[1]["val_mae"],
                    "test_mae": self.results[1]["test_mae"],
                    "hyperparameters": {"sigma": float("inf")},
                }
            )

        # Add regular training results
        for ntrain in sorted([k for k in self.results.keys() if k > 1]):
            result = self.results[ntrain]
            learning_curve_data.append(
                {
                    "ntrain": ntrain,
                    "val_rmse": result["val_rmse"],
                    "test_rmse": result["test_rmse"],
                    "val_mae": result["val_mae"],
                    "test_mae": result["test_mae"],
                    "hyperparameters": result["parameters"],
                    "eig_count": result["eig_count"],
                    "direct_count": result["direct_count"],
                }
            )

        self._archive["learning_curve"] = learning_curve_data
        self._archive["metadata"] = metadata

        with open(filename, "w") as f:
            json.dump(self._archive, f, indent=2)

        # Log what data was stored
        stored_sections = list(self._archive.keys())
        utils.info("Archive data stored", filename=filename, sections=stored_sections)

    def _create_holdout_split(self, elemental: bool = False):
        """Create training/holdout split based on max training size"""
        total_molecules = len(self.dataset)
        max_training_size = max(self._training_sizes)
        if max_training_size >= total_molecules:
            utils.error(
                "Max training size too large",
                max_training_size=max_training_size,
                total_molecules=total_molecules,
            )

        X_all = self.dataset.representations
        y_all = self.dataset.labels

        X_train = X_all[:max_training_size]
        self._y_train = y_all[:max_training_size]

        self._holdout_representer = X_all
        self._holdout_offset = max_training_size
        self._y_holdout = y_all[max_training_size:]

        self._local = X_train[0].ndim == 2 if X_train else False

        if self._local:
            self._train_counts = np.array([rep.shape[0] for rep in X_train])
            self._X_train = np.concatenate(X_train, axis=0)
            if elemental:
                charges_all = self.dataset.nuclear_charges
                self._train_nuclear_charges = np.concatenate(
                    charges_all[:max_training_size]
                )
        else:
            self._X_train = np.stack(X_train, axis=0)

        if self._detrend_atomic:
            element_counts, self._elements_Z = self.dataset.get_element_counts()
            self._elements_train = element_counts[:max_training_size]
            self._elements_holdout = element_counts[max_training_size:]

        if self._detrend_pairs:
            pair_features, self._pairs_labels = self.dataset.get_pairwise_features(
                self._detrend_pairs
            )
            self._pairs_train = pair_features[:max_training_size]
            self._pairs_holdout = pair_features[max_training_size:]

    def _detrend_matrix(
        self, is_train: bool, n: int | None = None
    ) -> np.ndarray | None:
        """Build joint detrending design matrix from active detrending modes.

        Args:
            is_train: True for training split, False for holdout.
            n: Number of training samples to use (only used when is_train=True).

        Returns:
            Design matrix of shape (n_molecules, n_features), or None if no
            detrending is active.
        """
        parts = []
        if self._detrend_atomic:
            A = self._elements_train[:n] if is_train else self._elements_holdout
            parts.append(A)
        if self._detrend_pairs:
            P = self._pairs_train[:n] if is_train else self._pairs_holdout
            parts.append(P)
        if not parts:
            return None
        return np.hstack(parts)

    def get_hyperparameter_grid(self, ntrain: int):
        factors = 2.0 ** np.arange(-1, 20)
        lam_grid = 10.0 ** np.arange(-10, -1)
        return factors, lam_grid

    def validation_size(self, ntrain: int) -> int:
        # default: 20%
        valcount = int(ntrain * 0.2)

        # if too large, far from ntrain, if too small, noisy
        return min(valcount, 200)

    def _optimize_hyperparameters(
        self, ntrain: int, length_heuristic: float
    ) -> tuple[float, float, float, int, int]:
        # other tricks which are not used yet:
        # when shuffling, in-group shuffles (validation vs training) could be ignored
        # cholesky updates
        # condition numbers could be estimated without full eigenvalue decomposition
        opt_start = time.time()
        best_params, best_val_rmse, best_val_mae = None, np.inf, None

        # Loop: sigma outer, splits inner
        factors, lam_grid = self.get_hyperparameter_grid(ntrain)
        shufs = 50
        validation = self.validation_size(ntrain)

        idx = np.arange(ntrain)

        y = self._y_train[:ntrain].copy()
        A = self._detrend_matrix(is_train=True, n=ntrain)
        if A is not None:
            coefs = linalg.lstsq(A, y)[0]
            if self._detrend_atomic:
                n_atomic = len(self._elements_Z)
                mapping = {
                    utils.Z_to_element_symbol(Z): float(c)
                    for Z, c in zip(self._elements_Z, coefs[:n_atomic])
                }
                utils.info("Atomic detrending coefficients", **mapping)
            if self._detrend_pairs:
                n_atomic = len(self._elements_Z) if self._detrend_atomic else 0
                for label, coef in zip(self._pairs_labels, coefs[n_atomic:]):
                    utils.info(
                        "Pairwise detrending coefficient", label=label, coef=float(coef)
                    )
            y -= A @ coefs
        y -= np.mean(y)
        # counter
        eig_count = 0
        direct_count = 0
        for factor in factors:
            # get kernel matrix
            sigma = length_heuristic * factor
            K_full = self._kernel_matrix.compute_train_kernel_matrix(sigma, ntrain)
            # kernel centering
            K_row_mean = K_full.mean(axis=1, keepdims=True)
            K_col_mean = K_full.mean(axis=0, keepdims=True)
            K_mean = K_full.mean()
            K_full = K_full - K_row_mean - K_col_mean + K_mean

            # choose algorithm based on condition number
            eigvals, Q = np.linalg.eigh(K_full)
            condition_number = eigvals[-1] / eigvals[1]
            if condition_number > 1e15:
                continue
            if condition_number < 5e8 and ntrain > 64:
                mode = "eig"
                eig_count += 1
            else:
                mode = "direct"
                direct_count += 1
            useschur = False
            if condition_number < 5e8 and ntrain > 128:
                useschur = True

            for lam_idx in range(len(lam_grid)):
                lam = lam_grid[lam_idx]

                if useschur:
                    diag_L = 1.0 / (eigvals + lam)

                split_rmse = []
                split_mae = []
                split_train_rmse = []
                split_train_mae = []
                for shuf_idx in range(shufs):
                    np.random.shuffle(idx)
                    y_shuf = y[idx]
                    if useschur:
                        Q_train = Q[idx[:-validation], :]
                        Q_val = Q[idx[-validation:], :]
                        y_train = y_shuf[:-validation]
                        H = (Q_val * diag_L) @ Q_val.T
                        L_Qt_y = diag_L * (Q_train.T @ y_train)
                        E_v = Q_train @ L_Qt_y
                        G_v = Q_val @ L_Qt_y
                        x = np.linalg.solve(H, G_v)
                        alpha = E_v - Q_train @ (diag_L * (Q_val.T @ x))
                    else:
                        K_full_shuf = K_full[idx][:, idx]
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", LinAlgWarning)
                                alpha = linalg.solve(
                                    K_full_shuf[:-validation, :-validation]
                                    + lam * np.eye(ntrain - validation),
                                    y_shuf[:-validation],
                                    assume_a="pos",
                                )
                        except linalg.LinAlgError:
                            continue

                    # validation
                    pred = K_full[idx[-validation:]][:, idx[:-validation]] @ alpha
                    rmse = np.sqrt(((pred - y_shuf[-validation:]) ** 2).mean())
                    mae = np.abs(pred - y_shuf[-validation:]).mean()
                    split_rmse.append(rmse)
                    split_mae.append(mae)

                    # training
                    pred_train = K_full[idx[:-validation]][:, idx[:-validation]] @ alpha
                    rmse_train = np.sqrt(
                        ((pred_train - y_shuf[:-validation]) ** 2).mean()
                    )
                    mae_train = np.abs(pred_train - y_shuf[:-validation]).mean()
                    split_train_rmse.append(rmse_train)
                    split_train_mae.append(mae_train)

                    if len(split_rmse) > 5:
                        one = np.median(split_rmse[::2])
                        two = np.median(split_rmse[1::2])
                        if abs(one - two) / np.median(split_rmse) < 5e-2:
                            break

                if len(split_rmse) < 5:
                    continue

                self._archive["hyperopt"].append(
                    {
                        "ntrain": ntrain,
                        "sigma": sigma,
                        "lambda": lam,
                        "val_rmse": split_rmse,
                        "val_mae": split_mae,
                        "train_rmse": split_train_rmse,
                        "train_mae": split_train_mae,
                        "eig_count": eig_count,
                        "direct_count": direct_count,
                    }
                )

                avg_rmse = np.median(split_rmse)
                avg_mae = np.median(split_mae)

                if avg_rmse < best_val_rmse:
                    best_val_rmse = avg_rmse
                    best_val_mae = avg_mae
                    best_params = {"sigma": sigma, "lambda": lam}

        opt_end = time.time()
        utils.info(
            "Hyperparameter optimization",
            ntrain=ntrain,
            duration=f"{opt_end - opt_start:.1f}s",
        )
        return best_params, best_val_rmse, best_val_mae, eig_count, direct_count

    def _evaluate_models(
        self,
        best_cases: dict[int, dict[str, float]],
    ) -> tuple[float, float]:
        models = {}
        y_tests = {}
        train_col_means = {}
        train_means = {}
        for ntrain, params in best_cases.items():
            y_train = self._y_train[:ntrain].copy()
            y_test = self._y_holdout.copy()
            A_train = self._detrend_matrix(is_train=True, n=ntrain)
            if A_train is not None:
                coefs = linalg.lstsq(A_train, y_train)[0]
                y_train -= A_train @ coefs
                y_test -= self._detrend_matrix(is_train=False) @ coefs

            shift = np.mean(y_train)
            y_train -= shift
            y_test -= shift

            K_train = self._kernel_matrix.compute_train_kernel_matrix(
                params["sigma"], ntrain
            )

            # Center the training kernel
            K_train_row_mean = K_train.mean(axis=1, keepdims=True)
            K_train_col_mean = K_train.mean(axis=0, keepdims=True)
            K_train_mean = K_train.mean()
            K_train_centered = (
                K_train - K_train_row_mean - K_train_col_mean + K_train_mean
            )
            # #Save means for the test kernel centering
            train_col_means[ntrain] = K_train_col_mean
            train_means[ntrain] = K_train_mean
            # #store eigenvalues for analysis
            w = np.linalg.eigvalsh(K_train_centered)
            self._archive["spectrum"] = self._archive.get("spectrum", {})
            self._archive["spectrum"][ntrain] = w.tolist()

            alpha = np.linalg.solve(
                K_train_centered + params["lambda"] * np.eye(len(y_train)), y_train
            )
            models[ntrain] = alpha
            y_tests[ntrain] = y_test

        model_preds = {_: list() for _ in models.keys()}
        # batched prediction to save memory: materialize one batch of holdout
        # representations at a time rather than keeping them all in memory
        n_holdout = len(self._y_holdout)
        batch_size = self._kernel_matrix._batch_size
        for ntrain, alpha in models.items():
            params_ntrain = best_cases[ntrain]
            K_train_col_mean = train_col_means[ntrain]
            K_train_mean = train_means[ntrain]
            for start in range(0, n_holdout, batch_size):
                end = min(start + batch_size, n_holdout)
                mols = self._holdout_representer._molecules[
                    self._holdout_offset + start : self._holdout_offset + end
                ]
                reps = self._holdout_representer.compute(mols)
                if self._local:
                    counts_batch = np.array([r.shape[0] for r in reps])
                    X_batch = np.concatenate(reps, axis=0)
                    nc_batch = (
                        np.concatenate([mol.get_atomic_numbers() for mol in mols])
                        if self._elemental
                        else None
                    )
                else:
                    X_batch = np.stack(reps, axis=0)
                    counts_batch = None
                    nc_batch = None
                K_test = self._kernel_matrix.compute_test_kernel_matrix(
                    params_ntrain["sigma"], ntrain, X_batch, counts_batch, nc_batch
                )
                K_test_row_mean = K_test.mean(axis=1, keepdims=True)
                K_test_centered = (
                    K_test - K_test_row_mean - K_train_col_mean + K_train_mean
                )
                model_preds[ntrain].append(K_test_centered @ alpha)

        for ntrain, preds in model_preds.items():
            pred = np.concatenate(preds, axis=0)
            y_test = y_tests[ntrain]

            # Store holdout predictions for residual calculation
            residuals = y_test - pred
            self.holdout_residuals[ntrain] = residuals

            test_rmse = np.sqrt(((pred - y_test) ** 2).mean())
            test_mae = np.abs(pred - y_test).mean()
            self.results[ntrain]["test_rmse"] = float(test_rmse)
            self.results[ntrain]["test_mae"] = float(test_mae)

    def _add_nullmodel(self) -> None:
        """Add nullmodel results where prediction is always the mean of the labels

        Validation metrics computed on training data, test metrics on holdout data.
        """
        y_train = self._y_train.copy()
        y_holdout = self._y_holdout.copy()
        A_train = self._detrend_matrix(is_train=True)
        if A_train is not None:
            coefs = linalg.lstsq(A_train, y_train)[0]
            y_train -= A_train @ coefs
            y_holdout -= self._detrend_matrix(is_train=False) @ coefs

        mean_prediction = np.mean(y_train)

        val_rmse = np.sqrt(((mean_prediction - y_train) ** 2).mean())
        val_mae = np.abs(mean_prediction - y_train).mean()

        test_rmse = np.sqrt(((mean_prediction - y_holdout) ** 2).mean())
        test_mae = np.abs(mean_prediction - y_holdout).mean()

        utils.info("Nullmodel results", test_rmse=float(test_rmse))
        self.results[1] = {
            "sigma_opt": np.inf,
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
        }
