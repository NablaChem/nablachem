import pandas as pd
import numpy as np

"""
Module: Hessian.py
------------------

This module estimates the **intrinsic dimensionality (ID)** of molecules in chemical space
based on the Taylor expansion of a scalar property (e.g., energy, HOMO-LUMO gap) using
gradients and Hessians with respect to 4N atomic coordinates: Z, x, y, z.

Key Concepts:
-------------
- ID quantifies how many effective degrees of freedom are needed to estimate a property.
- The Taylor expansion is evaluated within a **thermally accessible region** determined by
  energy-based bounds (KT = 4.75e-4).
- Eigenvalues and eigenvectors of the Hessian of the property are used to estimate the ID.
- The error between the full and truncated Hessian matrix helps determine ID.
- Degeneracy and gradient alignment checks refine ID estimate.

"""


class Estimator:
    """
    Estimate the intrinsic dimensionality (ID) of a scalar property in chemical space.

    Parameters
    ----------
    gradient_energy : array-like
        Gradient of the energy with respect to 4N atomic coordinates (Z, x, y, z).
    hessian_energy : array-like
        Hessian of the energy with respect to 4N atomic coordinates.
    gradient_proprety : array-like
        Gradient of the scalar property to be approximated (can be energy or another property).
    hessian_proprety : array-like
        Hessian of the scalar property to be approximated.
    dt : float
        Threshold used to detect degeneracy between eigenvalues.
    scaling_groups : list of bool, optional
        Boolean mask indicating which variables belong to chemical coordinates (True) versus
        spatial coordinates (False). Using the mask, both groups of the chemical coordinates
        and spatial ones can be scaled separately (it is allowed because are originated from
        different physical properties.). The porpuse is that to make the eigenvalues of the
        Hessiam matrix degenerate.
    """

    def __init__(
        self,
        gradient_energy,
        hessian_energy,
        gradient_proprety,
        hessian_proprety,
        dt,
        scaling_groups=None,
    ):
        self._g_e = np.array(gradient_energy)
        self._h_e = np.array(hessian_energy)
        self._g_p = np.array(gradient_proprety)
        self._h_p = np.array(hessian_proprety)
        self._n = self._g_e.shape[0]
        self._dt = dt
        self._scaling_groups = scaling_groups or [False] * self._n

    def _my_taylor(self, g, h, x):
        """
        Evaluate the second-order Taylor expansion of a scalar field at displacement x.

        Parameters
        ----------
        g : ndarray
            Gradient vector.
        h : ndarray
            Hessian matrix.
        x : ndarray
            Displacement vector.

        Returns
        -------
        float
            Value of the Taylor approximation.
        """
        sub_arrays = [x[i::4] for i in range(4)]
        reordered_array = np.concatenate(
            (sub_arrays[2], sub_arrays[0], sub_arrays[1], sub_arrays[3])
        )
        return np.dot(g, reordered_array) + 0.5 * np.dot(
            reordered_array.T, np.dot(h, reordered_array)
        )

    def get_bounds_analitical(self, mod_vec, g, h):
        """
        Determine thermal-accessible bounds for each eigenvector using energy.

        Parameters
        ----------
        mod_vec : ndarray
            Eigenvectors of the Hessian.
        g : ndarray
            Energy gradient.
        h : ndarray
            Energy Hessian.

        Returns
        -------
        list of list
            Bounds (positive and negative) for each eigenvector direction.
        """
        k = len(mod_vec)
        KT = 4.75e-4
        x = np.logspace(-10, 1, 100)
        bounds = []
        for i in range(k):
            q = []
            for sign in [1, -1]:
                for j in range(len(x)):
                    alpha = mod_vec[i] * sign * x[j]
                    if j == len(x) - 1:
                        q.append(sign * x[j])
                        break
                    if abs(self._my_taylor(g, h, alpha)) > KT:
                        q.append(sign * x[j])
                        break
            bounds.append(q)
        return bounds

    def getID(self):
        """
        Compute the intrinsic dimensionality (ID) by incrementally selecting eigenvectors
        that minimize the property estimation error.

        Returns
        -------
        dict
            Dictionary with:
            - "ID": List of dimensionality values (after degeneracy correction).
            - "Error": Corresponding estimation errors.
            - "natoms": Number of atoms (N).
        """
        Estimated_ID = []
        Estimated_degeneracy = [0]
        kept = []
        Estimated_ID.append(len(kept))

        alleigenvalues, alleigenvectors = np.linalg.eigh(self._h_p)
        bounds = self.get_bounds_analitical(alleigenvectors, self._g_e, self._h_e)
        errors = [self._estimate_error(alleigenvalues, bounds, kept)]

        if (
            np.all(self._g_e == self._g_e.flat[0])
            or np.all(self._h_e[0] == self._h_e[0].flat[0])
            or np.all(self._g_p == self._g_p.flat[0])
            or np.all(self._h_p[0] == self._h_p[0].flat[0])
        ):
            raise NotImplementedError(
                "All the values in the Hessian or gradient are the same"
            )

        while len(kept) < self._n:
            best_candidate = None
            best_error = 1
            for candidate in range(0, self._n):
                if candidate in kept:
                    continue
                new_kept = kept + [candidate]
                new_error = self._estimate_error(alleigenvalues, bounds, new_kept)
                if new_error < best_error:
                    best_candidate = candidate
                    best_error = new_error

            kept.append(best_candidate)
            mask = np.isin(np.arange(len(alleigenvalues)), np.array(kept))
            removed_indices = np.arange(len(alleigenvalues))[~mask]

            degeneracy = self._count_degeneracy(
                removed_indices, alleigenvalues, alleigenvectors
            )
            if (
                self._gradient_check(removed_indices, alleigenvalues, alleigenvectors)
                >= 0.95
            ):
                Estimated_ID.append(len(kept))
            else:
                Estimated_ID.append(len(kept) + 1)

            Estimated_degeneracy.append(degeneracy)
            errors.append(best_error)

            final_id = [
                Estimated_ID[_] - Estimated_degeneracy[_]
                for _ in range(len(Estimated_ID))
            ]

            d = {"ID": final_id, "Error": errors, "natoms": self._n // 4}
            df = pd.DataFrame(d)
            min_error_df = df.groupby("ID")["Error"].min().reset_index()
            result = {
                "ID": min_error_df["ID"].tolist(),
                "Error": min_error_df["Error"].tolist(),
                "natoms": d["natoms"],
            }

            new_ids = list(range(min(result["ID"]), max(result["ID"]) + 1))
            new_errors = []
            id_error_map = dict(zip(result["ID"], result["Error"]))
            last_error = 0
            for id_ in new_ids:
                if id_ in id_error_map:
                    last_error = id_error_map[id_]
                    new_errors.append(last_error)
                else:
                    new_errors.append(last_error)

            imputed_record = {"ID": new_ids, "Error": new_errors, "natoms": self._n / 4}
            id_collected = [_ for _ in range(self._n + 1)]
            error_collected = np.zeros(self._n + 1)
            for i in range(len(imputed_record["ID"])):
                error_collected[i] = imputed_record["Error"][i]

            final_result = {
                "ID": id_collected,
                "Error": error_collected,
                "natoms": self._n / 4,
            }
        return final_result

    def _estimate_error(self, mod_values, bounds, kept):
        """
        Estimate squared error of reconstruction when only `kept` eigenvalues are included.

        Parameters
        ----------
        mod_values : ndarray
            Eigenvalues of the property Hessian.
        bounds : list of list
            Integration bounds for each eigenvalues.
        kept : list of int
            Indices of retained eigenmodes.

        Returns
        -------
        float
            Approximate estimated error.
        """
        k = len(mod_values)
        delta_Q = 0
        for i in range(k):
            if i in kept:
                continue
            for j in range(k):
                if j in kept:
                    continue
                if i == j:
                    delta_Q += (
                        ((mod_values[i] ** 2) / (bounds[i][0] - bounds[i][1]))
                        * ((bounds[i][0] ** 5 - bounds[i][1] ** 5))
                        / 5
                    )
                else:
                    delta_Q += (
                        (
                            (mod_values[i] * mod_values[j])
                            / (
                                (bounds[i][0] - bounds[i][1])
                                * (bounds[j][0] - bounds[j][1])
                            )
                        )
                        * (
                            (bounds[i][0] ** 3 - bounds[i][1] ** 3)
                            * (bounds[j][0] ** 3 - bounds[j][1] ** 3)
                        )
                        / 9
                    )
        return np.sqrt(delta_Q)

    def _apply_scaling(self, H, s):
        """
        Apply coordinate-group-based scaling to the Hessian.

        Parameters
        ----------
        H : ndarray
            Input Hessian matrix.
        s : float
            Scaling factor for selected coordinate group.

        Returns
        -------
        ndarray
            Scaled Hessian matrix.
        """
        svec = np.ones(self._n)
        svec[self._scaling_groups] = s
        hmod = np.outer(svec, svec) * H
        return hmod

    def _count_degeneracy(self, removed_indices, mod_values, mod_vec):
        """
        Count maximum number of nearly degenerate eigenvalues across scaling sweeps.

        Parameters
        ----------
        removed_indices : list of int
            Indices of discarded eigenvalues.
        mod_values : ndarray
            Eigenvalues.
        mod_vec : ndarray
            Eigenvectors.

        Returns
        -------
        int
            Maximum observed degeneracy.
        """
        mod_values_copy = mod_values.copy()
        degeneracy_l = []
        for index in removed_indices:
            mod_values_copy[index] = 0
        mask = np.sort(abs(mod_values_copy))[::-1] > 0
        Lambda = np.diag(mod_values_copy)
        new_H = mod_vec @ Lambda @ np.linalg.inv(mod_vec)
        for s in np.sort(np.append(np.linspace(0.01, 10, 100), 1)):
            hmod_scaled = self._apply_scaling(new_H, s)
            scaled_values, _ = np.linalg.eigh(hmod_scaled)
            sorted_EV = (np.sort(abs(scaled_values))[::-1])[mask]
            degeneracy = sum(abs(np.diff(sorted_EV)) < self._dt)
            degeneracy_l.append(degeneracy)
        return max(degeneracy_l)

    def _gradient_check(self, removed_indices, mod_values, mod_vec):
        """
        Estimate alignment between remaining eigenvectors and the property gradient.

        Parameters
        ----------
        removed_indices : list of int
            Indices of discarded modes.
        mod_values : ndarray
            Eigenvalues.
        mod_vec : ndarray
            Eigenvectors.

        Returns
        -------
        float
            Total projection of the normalized gradient onto selected eigenvectors.
        """
        mod_values_copy = mod_values.copy()
        mod_vec_copy = mod_vec.copy()
        for index in removed_indices:
            mod_values_copy[index] = 0
        mask = np.sort(abs(mod_values_copy))[::-1] > 0
        selected_vec = mod_vec_copy[mask]
        projecyed_gradient = 0
        for i in range(len(selected_vec)):
            projecyed_gradient += np.abs(
                np.dot(
                    self._g_p / np.linalg.norm(self._g_p),
                    selected_vec[i] / np.linalg.norm(selected_vec[i]),
                )
            )
        return projecyed_gradient

    def _data_imputation(self, data):
        """
        Perform linear forward-fill on missing error values to complete error curve.

        Parameters
        ----------
        data : list of dict
            List of result records containing 'ID' and 'Error'.

        Returns
        -------
        list of dict
            List of completed records with full ID range and imputed errors.
        """
        min_error_data = []
        for i in range(len(data)):
            df = pd.DataFrame(data[i])
            min_error_df = df.groupby("ID")["Error"].min().reset_index()
            result = {
                "ID": min_error_df["ID"].tolist(),
                "Error": min_error_df["Error"].tolist(),
                "natoms": data[i]["natoms"],
            }
            min_error_data.append(result)
        imputed_data = []
        for record in min_error_data:
            new_ids = list(range(min(record["ID"]), max(record["ID"]) + 1))
            new_errors = []
            id_error_map = dict(zip(record["ID"], record["Error"]))
            last_error = 0
            for id_ in new_ids:
                if id_ in id_error_map:
                    last_error = id_error_map[id_]
                    new_errors.append(last_error)
                else:
                    new_errors.append(last_error)
            imputed_record = {
                "ID": new_ids,
                "Error": new_errors,
                "natoms": record["natoms"],
            }
            imputed_data.append(imputed_record)
        return imputed_data
