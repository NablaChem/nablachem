import numpy as np
import time
import json
from scipy import linalg
from numpy.polynomial.chebyshev import Chebyshev
import utils
from dataset import DataSet


class KernelMatrix:
    """Base class for kernel matrix computation and management"""

    def __init__(
        self, X: np.ndarray, X_holdout: np.ndarray = None, kernel_func: callable = None
    ):
        """Initialize kernel matrix with training and optional holdout data

        Args:
            X: Training data
            X_holdout: Holdout/test data (optional)
            kernel_func: Kernel function to use (optional)
        """
        self._X = X
        self._X_holdout = X_holdout
        self._kernel_func = kernel_func

        # Compute distance matrices
        self._D2 = self._dist_squared(X)
        if X_holdout is not None:
            self._D2_test = self._dist_squared(X, X_holdout)

    @staticmethod
    def _dist_squared(X_one: np.ndarray, X_other: np.ndarray = None) -> np.ndarray:
        """Compute squared Euclidean distances between points"""
        X_one = np.array(X_one)
        X_one_norm_sq = np.sum(X_one**2, axis=1)
        if X_other is None:
            X_other_norm_sq = X_one_norm_sq
            X_other = X_one
        else:
            X_other = np.array(X_other)
            X_other_norm_sq = np.sum(X_other**2, axis=1)
        D2 = X_other_norm_sq[:, None] + X_one_norm_sq[None, :] - 2 * (X_other @ X_one.T)
        D2 = np.maximum(D2, 0.0)
        return D2

    def compute_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training kernel matrix for given sigma and training size"""
        raise NotImplementedError("Subclasses must implement compute_kernel_matrix")

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for given sigma and training size"""
        raise NotImplementedError(
            "Subclasses must implement compute_test_kernel_matrix"
        )


class LocalKernelMatrix(KernelMatrix):
    """Kernel matrix for local (atom-based) representations with approximation cache"""

    def __init__(
        self,
        X: np.ndarray,
        train_counts: np.ndarray,
        X_holdout: np.ndarray = None,
        holdout_counts: np.ndarray = None,
    ):
        """Initialize local kernel matrix

        Args:
            X: Concatenated atom representations
            train_counts: Number of atoms per training molecule
            X_holdout: Concatenated holdout atom representations (optional)
            holdout_counts: Number of atoms per holdout molecule (optional)
        """
        super().__init__(X, X_holdout)
        self._train_counts = train_counts
        self._holdout_counts = holdout_counts

        # Local approximation cache will be built on demand
        self._cache_built = False

        # Compute holdout self-distances if needed
        if X_holdout is not None and holdout_counts is not None:
            self._test_self = []
            start = 0
            for count in holdout_counts:
                end = start + count
                self._test_self.append(
                    self._dist_squared(X_holdout[start:end], X_holdout[start:end])
                )
                start = end
        self.build_cache(max_molecules=len(train_counts))
        self._approx_fail_sigma = dict()

    def build_cache(self, max_molecules: int):
        """Build local approximation cache for efficient kernel computation"""
        self._local_grid = 1.5 ** np.linspace(-15, 15, 100)
        self._local_ymax = 20.0

        # Chebyshev polynomial coefficients for exp approximation
        cheby_p = [
            1.2783333716342860e-01,
            -2.4252536276891104e-01,
            2.0716160177307505e-01,
            -1.5966072205968104e-01,
            1.1136516853726638e-01,
            -7.0568587229867946e-02,
            4.0796581307398473e-02,
            -2.1612689660989802e-02,
            1.0538815782012821e-02,
            -4.7505844097694584e-03,
            1.9877638444287539e-03,
            -7.7505672091777230e-04,
            2.8263905844468264e-04,
            -9.6722980858327177e-05,
            3.1159309411000033e-05,
            -9.4769211836987947e-06,
            2.7285817731910956e-06,
            -7.4564575189212374e-07,
            1.9431609391984766e-07,
            -5.0571492223751847e-08,
            1.1357243950084354e-08,
        ]

        P = Chebyshev(cheby_p, domain=[0, self._local_ymax])
        Q = P.convert(kind=np.polynomial.Polynomial)
        self._exp_coef = Q.coef

        # Build power moments cache
        nmols = max_molecules
        atoms_per_mol = self._train_counts[:nmols]
        grid = self._local_grid
        D2 = self._D2
        npowers = len(cheby_p)
        npairs = nmols * (nmols + 1) // 2

        power_moments = np.zeros((npairs, npowers, len(grid)), dtype=np.float64)
        pair_idx = 0
        for i in range(nmols):
            for j in range(i, nmols):
                x = D2[
                    sum(atoms_per_mol[:i]) : sum(atoms_per_mol[: i + 1]),
                    sum(atoms_per_mol[:j]) : sum(atoms_per_mol[: j + 1]),
                ].flatten()
                x = np.sort(x)

                cum_moments = np.zeros((len(cheby_p), len(x) + 1))
                cum_moments[0, 1:] = np.cumsum(np.ones_like(x))
                for k in range(1, len(cheby_p)):
                    cum_moments[k, 1:] = np.cumsum(x**k)

                select_indices = np.minimum(
                    np.searchsorted(x, grid, side="right"), len(x) - 1
                )

                power_moments[pair_idx, :, :] = cum_moments[:, select_indices]
                pair_idx += 1

        self._local_power_moments = power_moments
        self._cache_built = True

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        nentries = sum(self._train_counts[:ntrain])
        section = self._D2[:nentries, :nentries].copy()
        np.fill_diagonal(section, np.inf)
        nnvals = np.amin(section, axis=0)
        return np.median(nnvals) ** 0.5

    @staticmethod
    def aggregate_atomic_kernel(
        K_atom_AB: np.ndarray, counts_A: np.ndarray, counts_B: np.ndarray
    ) -> np.ndarray:
        """Aggregate atomic kernel matrix to molecular kernel matrix"""
        counts_A = np.asarray(counts_A)
        counts_B = np.asarray(counts_B)

        starts_A = np.concatenate(([0], np.cumsum(counts_A)))
        starts_B = np.concatenate(([0], np.cumsum(counts_B)))

        # 2D prefix sum with 1-based indexing
        S = np.zeros((K_atom_AB.shape[0] + 1, K_atom_AB.shape[1] + 1))
        S[1:, 1:] = K_atom_AB
        S = S.cumsum(axis=0).cumsum(axis=1)

        # block boundaries
        a0 = starts_A[:-1][:, None]  # (MA,1)
        a1 = starts_A[1:][:, None]  # (MA,1)
        b0 = starts_B[:-1][None, :]  # (1,MB)
        b1 = starts_B[1:][None, :]  # (1,MB)

        # rectangular block extraction
        Kp = S[a1, b1] - S[a0, b1] - S[a1, b0] + S[a0, b0]
        return Kp

    def compute_kernel_matrix(self, sigma, ntrain):
        # approx only pays off for larger ntrain
        if ntrain <= 128:
            return self.compute_kernel_matrix_exact(sigma, ntrain)

        failsigma = self._approx_fail_sigma.get(ntrain, None)
        if failsigma is not None and sigma >= failsigma:
            return self.compute_kernel_matrix_exact(sigma, ntrain)
        approx = self.compute_kernel_matrix_approx(sigma, ntrain)
        if approx is None:
            self._approx_fail_sigma[ntrain] = sigma
            res = self.compute_kernel_matrix_exact(sigma, ntrain)
            return res
        return approx

    def compute_kernel_matrix_approx(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training local kernel matrix using approximation"""
        if not self._cache_built:
            raise RuntimeError("Cache must be built before computing kernel matrix")

        q = sigma**2
        cutoff = np.searchsorted(self._local_grid, self._local_ymax * q) - 1
        cutoff = max(0, min(cutoff, len(self._local_grid) - 1))

        moments = self._local_power_moments[:, :, cutoff]
        triu = moments * self._exp_coef
        triu /= q ** np.arange(len(self._exp_coef))
        triu = np.sum(triu, axis=1)

        # build full K matrix
        nmols = len(self._train_counts)
        K = np.zeros((nmols, nmols))
        pair_idx = 0
        for i in range(nmols):
            for j in range(i, nmols):
                K[i, j] = triu[pair_idx]
                K[j, i] = K[i, j]
                pair_idx += 1

        K_sub = K[:ntrain, :ntrain]

        # normalize
        d = np.diag(K_sub)
        d_sqrt = np.sqrt(d)
        K_sub /= np.outer(d_sqrt, d_sqrt)

        if np.max(K_sub) > 1.0 + 1e-8 or np.min(K_sub) > 0.1:
            return None
        return K_sub

    def compute_kernel_matrix_exact(self, sigma: float, ntrain: int) -> np.ndarray:
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        K_atom = np.exp(-self._D2[:natoms, :natoms] / sigma**2)
        K_test = self.aggregate_atomic_kernel(K_atom, atom_counts_A, atom_counts_A)

        # Compute normalization factors
        # For training: get unnormalized diagonal first
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)
        K_atom_train = np.exp(-self._D2[:natoms, :natoms] / sigma**2)
        K_train_unnorm = self.aggregate_atomic_kernel(
            K_atom_train, atom_counts_A, atom_counts_A
        )
        d_train_sqrt = np.sqrt(np.diag(K_train_unnorm))

        # Apply normalization
        K_test /= np.outer(d_train_sqrt, d_train_sqrt)

        return K_test

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for local representations"""
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        atom_counts_A = self._train_counts[:ntrain]
        atom_counts_B = self._holdout_counts
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        K_atom = np.exp(-self._D2_test[:, :natoms] / sigma**2)
        K_test = self.aggregate_atomic_kernel(K_atom, atom_counts_B, atom_counts_A)

        # Compute normalization factors
        # For training: get unnormalized diagonal first
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)
        K_atom_train = np.exp(-self._D2[:natoms, :natoms] / sigma**2)
        K_train_unnorm = self.aggregate_atomic_kernel(
            K_atom_train, atom_counts_A, atom_counts_A
        )
        d_train_sqrt = np.sqrt(np.diag(K_train_unnorm))

        # For test: self-kernel diagonal
        d_test = np.sqrt(
            [
                np.exp(-test_self_dist / sigma**2).sum()
                for test_self_dist in self._test_self
            ]
        )

        # Apply normalization
        K_test /= np.outer(d_test, d_train_sqrt)

        return K_test


class GlobalKernelMatrix(KernelMatrix):
    """Kernel matrix for global (molecule-based) representations"""

    def compute_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training kernel matrix for global representations"""
        D2_train = self._D2[:ntrain, :ntrain]
        # K_train = np.exp(-D2_train / sigma**2)
        K_train = self._kernel_func(np.sqrt(D2_train) / sigma)
        return K_train

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for global representations"""
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        D2_test = self._D2_test[:, :ntrain]
        K_test = self._kernel_func(np.sqrt(D2_test) / sigma)
        return K_test

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        section = self._D2[:ntrain, :ntrain].copy()
        np.fill_diagonal(section, np.inf)
        nnvals = np.amin(section, axis=0)
        return np.median(nnvals) ** 0.5


class AutoKRR:

    def __init__(
        self,
        dataset: DataSet,
        mincount: int,
        maxcount: int,
        kernel_func: callable,
        execution_commands: dict, 
        detrend_atomic: bool = True,
        output_name: str = "archive",
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
            self._kernel_matrix = LocalKernelMatrix(
                self._X_train, self._train_counts, self._X_holdout, self._holdout_counts
            )
        else:
            self._kernel_matrix = GlobalKernelMatrix(
                self._X_train, self._X_holdout, kernel_func
            )

        learning_curve_start = time.time()
        last_rmse = None
        last_size = None
        factor_min = -10 
        factor_max =  20
        lambda_min = -14
        lambda_max = -1
        extended_range = 6
        speed_up_factors = []
        speed_up_lambdas = []

        for i, ntrain in enumerate(self._training_sizes):
            combinations_tested = (factor_max-factor_min) * (lambda_max-lambda_min)
            utils.info(
                f"Training size: {ntrain}",
                combinations_tested = combinations_tested,
                λ_range=(lambda_min,lambda_max),
                σ_range=(factor_min,factor_max),
            )

            length_heuristic = self._kernel_matrix.length_scale(ntrain)
            best_parameters, best_val_rmse, best_val_mae = (
                self._optimize_hyperparameters(ntrain, length_heuristic,(factor_min,factor_max),(lambda_min,lambda_max))
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
                "combinations_tested": int(combinations_tested),
                **improvement,
            }
            
            if ntrain >= 128 and ntrain <= 1024:
                factor_exp = np.log(best_parameters["sigma"] / length_heuristic) / np.log(1.5)
                speed_up_factors.append(int(round(factor_exp)))
                speed_up_lambdas.append(int(round(np.log10(best_parameters["lambda"]))))
                
            if ntrain == 1024:
                # New values are in a range between the lower and upper values from those best parameters with an expanded range 
                new_factor_min = min(speed_up_factors) - extended_range
                new_factor_max = max(speed_up_factors) + extended_range
                new_lambda_min = min(speed_up_lambdas) - extended_range
                new_lambda_max = max(speed_up_lambdas) + extended_range
                
                factor_min = max(factor_min,new_factor_min) 
                factor_max = min(factor_max,new_factor_max)
                lambda_min = max(lambda_min,new_lambda_min)
                lambda_max = min(lambda_max,new_lambda_max)
            
            self.store_archive(f"{output_name}.json",execution_commands)
            print()
        learning_curve_end = time.time()
        utils.info(
            "Learning curve calculation",
            duration=f"{learning_curve_end - learning_curve_start:.1f}s",
        )

    def store_archive(self, filename: str, execution_commands: dict) -> None:
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
                    "combinations_tested": self.results[1]["combinations_tested"]
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
                    "combinations_tested": result["combinations_tested"],
                }
            )

        self._archive["learning_curve"] = learning_curve_data
        self._archive["Commands"] = execution_commands

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
        self, ntrain: int, 
        length_heuristic: float,
        factors_range: tuple[float,float],
        lambdas_range: tuple[float,float],
    ) -> tuple[float, float, float]:
        # other tricks which are not used yet:
        # when shuffling, in-group shuffles (validation vs training) could be ignored
        # cholesky updates
        opt_start = time.time()
        best_params, best_val_rmse, best_val_mae = None, np.inf, None

        # Loop: sigma outer, splits inner
        factors = 1.5 ** np.arange(factors_range[0], factors_range[1])
        lam_grid = 10.0 ** np.arange(lambdas_range[0], lambdas_range[1])
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
            K_full = self._kernel_matrix.compute_kernel_matrix(sigma, ntrain)

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
            σ = float(best_params["sigma"]),
            λ = float(best_params["lambda"]),
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

        K_train = self._kernel_matrix.compute_kernel_matrix(params["sigma"], ntrain)
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
