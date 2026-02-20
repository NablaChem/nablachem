import numpy as np
import time
import json
from scipy import linalg
from . import utils
from . import matrix
from .dataset import DataSet


class AutoKRR:

    def __init__(
        self,
        dataset: DataSet,
        mincount: int,
        maxcount: int,
        kernel_func: callable,
        detrend_atomic: bool = True,
    ) -> None:
        self._archive = {}
        self._archive["hyperopt"] = []
        self.dataset = dataset
        self._training_sizes = utils.get_training_sizes(mincount, maxcount)
        self._detrend_atomic = detrend_atomic

        self._create_holdout_split()

        self.results: dict[int, dict[str, float]] = {}
        self.holdout_residuals: dict[int, np.ndarray] = {}
        self._add_nullmodel()

        if self._local:
            self._kernel_matrix = matrix.LocalKernelMatrix(
                self._X_train, self._train_counts, self._X_holdout, self._holdout_counts
            )
        else:
            self._kernel_matrix = matrix.GlobalKernelMatrix(
                self._X_train, self._X_holdout, kernel_func
            )

        learning_curve_start = time.time()
        last_rmse = None
        last_size = None
        for i, ntrain in enumerate(self._training_sizes):
            length_heuristic = self._kernel_matrix.length_scale(ntrain)
            best_parameters, best_val_rmse, best_val_mae = (
                self._optimize_hyperparameters(ntrain, length_heuristic)
            )
            test_rmse, test_mae = self._evaluate_model(ntrain, best_parameters)

            improvement = {}
            if last_rmse is not None:
                improvement["test_slope"] = float(
                    np.log(test_rmse / last_rmse) / np.log(ntrain / last_size)
                )
            else:
                improvement["test_slope"] = None

            last_rmse = test_rmse
            last_size = ntrain

            utils.info(
                "Training size completed",
                ntrain=ntrain,
                test_rmse=float(test_rmse),
                **improvement,
            )

            self.results[ntrain] = {
                "parameters": best_parameters,
                "val_rmse": float(best_val_rmse),
                "val_mae": float(best_val_mae),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                **improvement,
            }

        learning_curve_end = time.time()
        utils.info(
            "Learning curve calculation",
            duration=f"{learning_curve_end - learning_curve_start:.1f}s",
        )

    def store_archive(self, filename: str) -> None:
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
                }
            )

        self._archive["learning_curve"] = learning_curve_data

        with open(filename, "w") as f:
            json.dump(self._archive, f, indent=2)

        # Log what data was stored
        stored_sections = list(self._archive.keys())
        utils.info("Archive data stored", filename=filename, sections=stored_sections)

    def _create_holdout_split(self):
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

        self._X_holdout = X_all[max_training_size:]
        self._y_holdout = y_all[max_training_size:]

        self._local = X_train[0].ndim == 2 if X_train else False

        if self._local:
            self._train_counts = np.array([rep.shape[0] for rep in X_train])
            self._X_train = np.concatenate(X_train, axis=0)
            self._holdout_counts = np.array([rep.shape[0] for rep in self._X_holdout])
            self._X_holdout = np.concatenate(self._X_holdout, axis=0)
        else:
            self._X_train = np.stack(X_train, axis=0)
            self._X_holdout = np.stack(self._X_holdout, axis=0)

        if self._detrend_atomic:
            element_counts, self._elements_Z = self.dataset.get_element_counts()
            self._elements_train = element_counts[:max_training_size]
            self._elements_holdout = element_counts[max_training_size:]

    def _optimize_hyperparameters(
        self, ntrain: int, length_heuristic: float
    ) -> tuple[float, float, float]:
        # other tricks which are not used yet:
        # when shuffling, in-group shuffles (validation vs training) could be ignored
        # cholesky updates
        opt_start = time.time()
        best_params, best_val_rmse, best_val_mae = None, np.inf, None

        # Loop: sigma outer, splits inner
        factors = 1.5 ** np.arange(-10, 20)
        lam_grid = 10.0 ** np.arange(-14, -1)
        shufs = 20
        validation = 50

        idx = np.arange(ntrain)

        y = self._y_train[:ntrain].copy()
        if self._detrend_atomic:
            A = self._elements_train[:ntrain]
            coefs = linalg.lstsq(A, y)[0]
            mapping = {
                utils.Z_to_element_symbol(Z): float(c)
                for Z, c in zip(self._elements_Z, coefs)
            }
            utils.info("Atomic detrending coefficients", **mapping)
            trend = A @ coefs
            y -= trend
        y -= np.mean(y)

        for factor in factors:
            # get kernel matrix
            sigma = length_heuristic * factor
            K_full = self._kernel_matrix.compute_train_kernel_matrix(sigma, ntrain)

            # choose algorithm based on condition number
            eigvals, Q = np.linalg.eigh(K_full)
            condition_number = eigvals[-1] / eigvals[0]
            if condition_number > 1e15:
                continue
            if condition_number < 5e6 and ntrain > 64:
                mode = "eig"
            else:
                mode = "direct"
            useschur = False
            if condition_number < 1e7 and ntrain > 128:
                useschur = True

            for lam_idx in range(len(lam_grid)):
                lam = lam_grid[lam_idx]

                if mode == "eig" and useschur:
                    mid = 1.0 / (eigvals + lam)
                    Kinv = (Q * mid) @ Q.T
                if mode == "direct" and useschur:
                    Kinv = np.linalg.inv(K_full + lam * np.eye(ntrain))

                split_rmse = []
                split_mae = []
                split_train_rmse = []
                split_train_mae = []
                for shuf_idx in range(shufs):
                    np.random.shuffle(idx)
                    y_shuf = y[idx]
                    if useschur:
                        Kinv_shuf = Kinv[idx][:, idx]
                        E = Kinv_shuf[:-validation, :-validation]
                        H = Kinv_shuf[-validation:, -validation:]
                        F = Kinv_shuf[:-validation, -validation:]
                        G = Kinv_shuf[-validation:, :-validation]
                        H_inv = np.linalg.inv(H)
                        alphapart = E @ y_shuf[:-validation] - F @ (
                            H_inv @ (G @ y_shuf[:-validation])
                        )
                        alpha = alphapart
                    else:
                        K_full_shuf = K_full[idx][:, idx]
                        try:
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
        return best_params, best_val_rmse, best_val_mae

    def _evaluate_model(
        self,
        ntrain: int,
        params: dict[str, float],
    ) -> tuple[float, float]:
        y_train = self._y_train[:ntrain].copy()
        y_test = self._y_holdout.copy()
        if self._detrend_atomic:
            A = self._elements_train[:ntrain]
            coefs = linalg.lstsq(A, y_train)[0]
            trend_train = A @ coefs
            y_train -= trend_train

            A_test = self._elements_holdout
            trend_test = A_test @ coefs
            y_test -= trend_test

        shift = np.mean(y_train)
        y_train -= shift
        y_test -= shift

        K_train = self._kernel_matrix.compute_train_kernel_matrix(
            params["sigma"], ntrain
        )
        K_test = self._kernel_matrix.compute_test_kernel_matrix(params["sigma"], ntrain)

        # store eigenvalues for analysis
        w = np.linalg.eigvalsh(K_train)
        self._archive["spectrum"] = self._archive.get("spectrum", {})
        self._archive["spectrum"][ntrain] = w.tolist()

        alpha = np.linalg.solve(
            K_train + params["lambda"] * np.eye(len(y_train)), y_train
        )
        pred = K_test @ alpha

        # Store holdout predictions for residual calculation
        residuals = y_test - pred
        self.holdout_residuals[ntrain] = residuals

        test_rmse = np.sqrt(((pred - y_test) ** 2).mean())
        test_mae = np.abs(pred - y_test).mean()
        return test_rmse, test_mae

    def _add_nullmodel(self) -> None:
        """Add nullmodel results where prediction is always the mean of the labels

        Validation metrics computed on training data, test metrics on holdout data.
        """
        y_train = self._y_train.copy()
        y_holdout = self._y_holdout.copy()
        if self._detrend_atomic:
            A = self._elements_train
            coefs = linalg.lstsq(A, y_train)[0]
            trend = A @ coefs
            y_train -= trend
            A_holdout = self._elements_holdout
            trend_holdout = A_holdout @ coefs
            y_holdout -= trend_holdout

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
