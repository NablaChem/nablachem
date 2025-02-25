import pandas as pd
import findiff
import numpy as np
import math
import collections
import itertools as it
import pyscf
import enum

from scipy.optimize import minimize

from .analyticgrads.AP_class import APDFT_perturbator as AP
from .analyticgrads.AP_class import alchemy_cphf_deriv


class Monomial:
    """A single monomial in the multi-dimensional Taylor expansion."""

    def __init__(self, prefactor: float, powers: dict[str, int] = {}):
        """Define the monomial.

        Parameters
        ----------
        prefactor : float
            Weight or coefficient of the monomial.
        powers : dict[str, int], optional
            Involved variables as keys and the exponent as value, by default {}.
        """
        self._powers = powers
        self._prefactor = prefactor
        self._cached_prefactor = None

    def __repr__(self):
        return f"Monomial({self._prefactor}, {self._powers})"

    def prefactor(self) -> float:
        """Calculates the Taylor expansion prefactor.

        Returns
        -------
        float
            Prefactor for the summation in the Taylor expansion.
        """
        if self._cached_prefactor is None:
            self._cached_prefactor = self._prefactor / np.prod(
                [math.factorial(_) for _ in self._powers.values()]
            )
        return self._cached_prefactor

    def distance(self, pos: dict[str, float], center: dict[str, float]) -> float:
        """Evaluate the distance term of the Taylor expansion.

        Parameters
        ----------
        pos : dict[str, float]
            The position at which the Monomial is evaluated. Keys are the variable names, values are the positions.
        center : dict[str, float]
            The center of the Taylor expansion. Keys are the variable names, values are the positions.

        Returns
        -------
        float
            Distance
        """
        ret = []
        for column, power in self._powers.items():
            ret.append((pos[column] - center[column]) ** power)
        return math.prod(ret)


class MultiTaylor:
    """Multi-dimensional multi-variate arbitrary order Taylor expansion from any evenly spaced finite difference stencil.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("some_file.csv")
    >>> df.columns
    Index(['RX', 'RY', 'RZ', 'QX', 'QY', 'QZ', 'E', 'BETA1', 'BETA2',
       'SIGMA'],
      dtype='object')
    >>> mt = MultiTaylor(df, outputs="BETA1 BETA2 SIGMA".split())
    >>> spatial_center, electronic_center = 3, 2.5
    >>> mt.reset_center(
        RX=spatial_center,
        RY=spatial_center,
        RZ=spatial_center,
        QX=electronic_center,
        QY=electronic_center,
        QZ=electronic_center,
    )
    >>> mt.reset_filter(E=4)
    >>> mt.build_model(2)
    >>> mt.query(RX=3.1, RY=3.1, RZ=3.1, QX=2.4, QY=2.4, QZ=2.4)
    {'BETA1': 0.022412699999999976,
    'BETA2': 0.014047600000000134,
    'SIGMA': 0.0018744333333333316}
    """

    def __init__(self, dataframe: pd.DataFrame, outputs: list[str]):
        """Initialize the Taylor expansion from a dataframe of data points forming the superset of stencils.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Holds all data points available for the vicinity of the future center of the expansion.
        outputs : list[str]
            Those columns of the dataframe that are considered to be outputs rather than input coordinates.
        """
        self._dataframe = dataframe
        self._outputs = outputs
        self._filtered = dataframe
        self._filter = {}

    def reset_center(self, **kwargs: float):
        """Sets the expansion center from named arguments for each column."""

        self._center = kwargs

    def reset_filter(self, **kwargs: float):
        """Sets the filter for the dataframe from named arguments for each column.

        All columns which are not filtered and not outputs are considered to be input coordinates.
        """
        self._filtered = self._dict_filter(self._dataframe, kwargs)
        self._filter = kwargs

    def _dict_filter(self, df: pd.DataFrame, filter: dict[str, float]) -> pd.DataFrame:
        """Applies a filter to a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Stencil dataframe
        filter : dict[str, float]
            Filter to apply. Keys are the column names, values are the values to filter for.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        return df.loc[(df[list(filter)] == pd.Series(filter)).all(axis=1)]

    def _offsets(self) -> tuple[np.ndarray, tuple[str]]:
        """Transforms the available data points in the dataframe into stencil offsets.

        Returns
        -------
        tuple[np.ndarray, tuple[str]]
            Offsets for the stencil and list of variable names in matching order.
        """
        variables = [_ for _ in self._filtered.columns if _ not in self._outputs]
        center = np.array([self._center[_] for _ in variables])
        offsets = self._filtered[variables].values - center
        return offsets, tuple(variables)

    def to_constant_grad_and_hess(self, output: str):
        """Exports the Taylor expansion for a given output as constant, gradient, and Hessian.

        Parameters
        ----------
        output : str
            The output variable for which this analysis is done.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray, list[str]]
            Constant, gradient, Hessian, and ordered list of variable names.
        """
        args = list(sorted(self._center.keys()))
        nargs = len(args)
        constant = None
        grad = np.zeros(nargs)
        hess = np.zeros((nargs, nargs))
        for monomial in self._monomials[output]:
            coef = monomial._prefactor
            order = sum(monomial._powers.values())
            if order == 0:
                constant = coef
            elif order == 1:
                (a,) = monomial._powers.keys()
                idx = args.index(a)
                grad[idx] = coef
            elif order == 2:
                try:
                    a, b = monomial._powers.keys()
                except:
                    (a,) = monomial._powers.keys()
                    b = a
                idx1 = args.index(a)
                idx2 = args.index(b)
                hess[idx1, idx2] = coef
                hess[idx2, idx1] = coef
        return constant, np.array(grad), np.array(hess), args

    def _build_monomials(
        self, term: tuple[str], shifted: tuple[np.ndarray, tuple[str]]
    ):
        """Builds all monomials and ensures that the stencil is applicable.

        Parameters
        ----------
        term : str
            String-based term representation.
        shifted : tuple[np.ndarray, tuple[str]]:
            Cached version of self._offset()

        Raises
        ------
        ValueError
            Could not build stencil.
        ValueError
            Not enough points in the stencil for a given order.
        """
        offsets, ordering = shifted
        term_counter = collections.Counter(term)
        indices = tuple([term_counter[_] for _ in ordering])
        order = len(term)

        # shortcut for stencils similar to central differences
        # often it is sufficient to have one value per combination of changes
        # e.g. in 1D, select one point to the left, one to the right of the center
        # e.g. in 2D, select one point from every quadrant, plus the center
        shortcut = None
        mask_center = np.all(offsets == 0, axis=1)
        if len(set(term)) == 1:
            column_mask = offsets[0, :] != offsets[0, :]
            column_mask[ordering.index(term[0])] = True
            shortcut = (offsets[:, column_mask] != 0).reshape(-1) & np.all(
                offsets[:, ~column_mask] == 0, axis=1
            )
            shortcut |= mask_center
        if len(set(term)) > 0:
            n_distinct_columns = len(set(term))
            shortcut = mask_center
            column_mask = np.array([False] * offsets.shape[1])
            for column in term:
                column_mask[ordering.index(column)] = True
            other_columns_are_centered = np.all(offsets[:, ~column_mask] == 0, axis=1)
            for hypercube in it.product((False, True), repeat=n_distinct_columns):
                hypercube_mask = True
                for sign, column in zip(hypercube, term):
                    hypercube_mask &= (offsets[:, ordering.index(column)] > 0) == sign
                hypercube_mask &= other_columns_are_centered
                hypercube_mask &= ~mask_center

                # keep only a few from this hypercube
                # avoids slowdown because of very imbalanced data-sets
                # parameter "extra" is free to choose (>0), should be generous
                # choose closest to origin
                extra = 3
                all_selected = np.where(hypercube_mask)[0]
                distance = np.linalg.norm(offsets[hypercube_mask, :], axis=1)
                keep = order + extra
                if len(distance) > keep:
                    too_far = np.argpartition(distance, keep)[keep:]
                    too_far = all_selected[too_far]
                    hypercube_mask[too_far] = False

                shortcut |= hypercube_mask

            for this, count in term_counter.items():
                if count < 2:
                    continue

                column_mask[True] = True
                for column in term:
                    column_mask[ordering.index(column)] = False
                column_mask[ordering.index(this)] = True

                higher_order_mask = np.all(offsets[:, column_mask] == 0, axis=1)
                shortcut |= higher_order_mask

        # auxiliary heuristics for viable subsets
        relevant_indices = [i for i, v in enumerate(indices) if v != 0]
        mask_changed = ~np.all(offsets[:, relevant_indices] == 0, axis=1)
        base_mask = mask_changed | mask_center

        mask_up_to_order = np.sum(offsets != 0, axis=1) <= order

        # try with simple masks first
        tries = [
            shortcut,
            base_mask & mask_up_to_order,
            base_mask,
            base_mask | ~base_mask,
        ]
        for mask in tries:
            subset_offsets = offsets[mask, :]

            # condense: drop all columns which are all zeros
            non_zero_columns = np.where(~np.all(subset_offsets == 0, axis=0))[0]
            subset_offsets = subset_offsets[:, non_zero_columns]

            # make sure that no non-zero index is dropped
            abort_mask = False
            for i in range(len(indices)):
                if indices[i] > 0 and i not in non_zero_columns:
                    abort_mask = True
                    break
            if abort_mask:
                continue

            mod_indices = tuple([indices[_] for _ in non_zero_columns])

            # for better numerical stability, try different scalings
            optimal_stencil = None
            for stencil_scaling in (1, 10, 100, 1000):
                scaled_subset_offsets = tuple(
                    map(tuple, subset_offsets * stencil_scaling)
                )
                try:
                    stencil = findiff.stencils.Stencil(
                        scaled_subset_offsets,
                        partials={mod_indices: 1},
                        spacings=1 / stencil_scaling,
                    )
                except:
                    # Numerical issue -> next try
                    continue

                # Did not find a stencil -> next try
                if len(stencil.values) == 0:
                    continue

                # usually the one with the fewest points has the least noise
                if optimal_stencil is None or len(optimal_stencil) > len(
                    stencil.values
                ):
                    optimal_stencil = stencil.values
                    optimal_offsets = scaled_subset_offsets

            if optimal_stencil is None:
                continue

            for output in self._outputs:
                weights = [
                    optimal_stencil[_] if _ in optimal_stencil else 0
                    for _ in optimal_offsets
                ]
                values = self._filtered[output].values[mask]

                self._monomials[output].append(
                    Monomial(
                        prefactor=np.dot(weights, values),
                        powers=dict(term_counter),
                    )
                )
            return
        raise ValueError(f"Could not build stencil for term {term}.")

    @property
    def _data_columns(self):
        data_columns = [_ for _ in self._filtered.columns if _ not in self._outputs]
        data_columns = [_ for _ in data_columns if not _ in self._filter.keys()]
        return data_columns

    def _all_terms_up_to(self, order: int) -> tuple[tuple[str]]:
        """For all remaining input columns, find all possible terms entering a Taylor expansion.

        Parameters
        ----------
        order : int
            The maximum order of the expansion.

        Returns
        -------
        tuple[tuple[str]]
            Series of terms up to the given order, as a tuple of variable names.
        """
        terms = []

        for order in range(1, order + 1):
            for entry in it.combinations_with_replacement(self._data_columns, order):
                terms.append(entry)
        return tuple(terms)

    def build_model(self, orders: int, additional_terms: list[tuple[str]] = []):
        """Sets up the model for a specific expansion order or list of terms.

        Parameters
        ----------
        orders : int
            All terms are included in the expansion up to this order.
        additional_terms : list[tuple[str]]
            The terms to ADDITIONALLY include, i.e. list of tuples of column names.

            To only include d/dx, give [('x',)]
            To only include d^2/dx^2, give [('x', 'x')]
            To only include d^2/dxdy, give [('x', 'y')]
            To include all three, give [('x',), ('x', 'x'), ('x', 'y')]

        Raises
        ------
        NotImplementedError
            Center needs to be given in dataframe.
        ValueError
            Center is not unique.
        ValueError
            Duplicate points in the dataset.
        ValueError
            Invalid column names for additonal terms.
        """
        # check center: there can be only one
        center_rows = self._dict_filter(self._dataframe, self._center)
        center_row = self._dict_filter(center_rows, self._filter)
        if len(center_row) == 0:
            raise NotImplementedError(f"Center is not in the dataframe.")
        if len(center_row) > 1:
            raise ValueError(f"Center is not unique.")

        # check for duplicates
        shifted = self._offsets()
        if len(self._filtered[list(shifted[1])].drop_duplicates()) != len(
            self._filtered
        ):
            raise ValueError(f"Duplicate points in the dataset.")

        # setup constant term
        self._monomials = {k: [Monomial(center_row.iloc[0][k])] for k in self._outputs}

        # check for valid names
        for term in additional_terms:
            if len(set(term) - set(self._data_columns)) > 0:
                raise ValueError(
                    f"Invalid column name in {term}, needs to be in {self._data_columns}."
                )

        terms = tuple(list(self._all_terms_up_to(orders)) + list(additional_terms))
        if len(terms) > len(shifted[0]):
            raise ValueError(
                f"Not enough points: {len(terms)} required, {len(shifted[0])} given."
            )
        for term in terms:
            self._build_monomials(term, shifted)

    def query(self, **kwargs: float) -> float:
        """Evaluate the Taylor expansion at a given point.

        Returns
        -------
        float
            Value from all terms.
        """
        ret = {}
        for output in self._outputs:
            ret[output] = 0
            for monomial in self._monomials[output]:
                prefactor = monomial.prefactor()
                ret[output] += prefactor * monomial.distance(kwargs, self._center)
        return ret

    def query_detail(
        self, output: str, **kwargs: float
    ) -> dict[tuple[str, int], float]:
        """Breaks down the Taylor expansion into its monomials.

        Parameters
        ----------
        output : str
            The output variable for which this analysis is done.

        Returns
        -------
        dict[tuple[str, int], float]
            Keys are the variable names and the exponents, values are the contributions from each monomial.
        """
        ret = {}
        for monomial in self._monomials[output]:
            prefactor = monomial.prefactor()
            ret[tuple(monomial._powers.items())] = prefactor * monomial.distance(
                kwargs, self._center
            )
        return ret

    def _optimize(
        self, maximize: bool, target: str, bounds: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        """Optimizes the target value over the the input variables within bounds.

        Parameters
        ----------
        maximize : bool
            Whether to maximize.
        target : str
            Column name to optimize.
        bounds : dict[str, tuple[float, float]]
            Keys are the variable names, values are the bounds.

        Returns
        -------
        dict[str, float]
            Best point found. Keys are the variable names, values are the positions.
        """
        ordering = bounds.keys()
        sign = -1 if maximize else 1

        def f(x):
            point = {k: v for k, v in zip(ordering, x)}
            return sign * self.query(**point)[target]

        res = minimize(
            f,
            x0=[self._center[_] for _ in ordering],
            bounds=[bounds[_] for _ in ordering],
            method="L-BFGS-B",
            tol=1e-6,
            options={"maxiter": 1000},
        )
        return {k: v for k, v in zip(ordering, res.x)}

    def minimize(
        self, target: str, bounds: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        """See _optimize.

        Parameters
        ----------
        target : str
            Column name to minimize.
        bounds : dict[str, tuple[float, float]]
            Bounds for the search space.

        Returns
        -------
        dict[str, float]
            Optimal position found.
        """
        return self._optimize(False, target, bounds)

    def maximize(
        self, target: str, bounds: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        """See _optimize.

        Parameters
        ----------
        target : str
            Column name to maximize.
        bounds : dict[str, tuple[float, float]]
            Bounds for the search space.

        Returns
        -------
        dict[str, float]
            Optimal position found.
        """
        return self._optimize(True, target, bounds)


class Anygrad:
    """Calculates quantum chemical gradients including those w.r.t. nuclear charges analytically where possible.

    Order: xyzZxyzZxyzZ... (in order of atoms)
    """

    _references = {"CP-RHF": "DOI 10.1063/5.0085817"}

    def _energy_R_RHF_CP(self, calculator=None):
        if calculator is None:
            calculator = self._calculator
        calculator.kernel()
        grad = calculator.Gradients().kernel()
        return grad.reshape(3 * self._natm)

    def _energy_R_R_RHF_CP(self):
        self._calculator.kernel()
        hess = self._calculator.Hessian().kernel()
        return hess.transpose(0, 2, 1, 3).reshape(3 * self._natm, -1)

    def _energy_Z_RHF_CP(self, calculator=None):
        if calculator is None:
            calculator = self._calculator
        ap = AP(calculator, sites=range(self._natm))
        return ap.build_gradient()

    def _homo_Z_RHF_CP(self, calculator=None):
        if calculator is None:
            calculator = self._calculator

        mocc = calculator.mo_occ > 0
        homo_idx = calculator.mo_energy[mocc].argmax()

        depsilon = []
        for atomidx in range(self._natm):
            calculator.mol.set_rinv_orig_(calculator.mol.atom_coords()[atomidx])
            dV = -calculator.mol.intor("int1e_rinv")
            _, e1 = alchemy_cphf_deriv(calculator, dV)
            depsilon.append(np.diag(e1)[homo_idx])

        return np.array(depsilon)

    def _homo_R_RHF_CP(self, calculator=None):
        if calculator is None:
            calculator = self._calculator
        from pyscf import hessian

        mf_h = self._calculator.Hessian()
        h1 = hessian.rhf.make_h1(mf_h, calculator.mo_coeff, calculator.mo_occ)

        _, mo_e1 = hessian.rhf.solve_mo1(
            calculator,
            calculator.mo_energy,
            calculator.mo_coeff,
            calculator.mo_occ,
            h1,
            None,
        )
        return np.asarray(mo_e1)[:, :, -1, -1].reshape(-1)

    def _energy_Z_Z_RHF_CP(self):
        ap = AP(self._calculator, sites=range(self._natm))
        return ap.build_hessian()

    def _energy_R_RKS_CP(self):
        return self._energy_R_RHF_CP()

    def _energy_R_R_RKS_CP(self):
        return self._energy_R_R_RHF_CP()

    def _energy_Z_RKS_CP(self):
        return self._energy_Z_RHF_CP()

    def _energy_Z_Z_RKS_CP(self):
        return self._energy_Z_Z_RHF_CP()

    def _energy_R_Z_RHF_CP(self):
        sites = range(self._natm)
        ap = AP(self._calculator, sites=sites)
        return np.array([ap.af(i).reshape(-1) for i in sites]).T

    def _energy_Z_R_RHF_CP(self):
        return self._energy_R_Z_RHF_CP().T

    def _energy_R_RHF_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, :3].reshape(-1)

    def _energy_R_RKS_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RKS", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, :3].reshape(-1)

    def _energy_Z_RKS_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RKS", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, 3].reshape(-1)

    def _energy_Z_RHF_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, 3].reshape(-1)

    def _energy_R_R_RHF_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RHF", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask][:, mask]

    def _energy_Z_Z_RHF_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][2][3::4, 3::4]

    def _energy_R_Z_RHF_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RHF", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask, 3::4]

    def _energy_Z_R_RHF_FD(self):
        return self._energy_R_Z_RHF_FD().T

    def _energy_R_R_RKS_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RKS", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask][:, mask]

    def _energy_Z_Z_RKS_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RKS", target)
        return self._fd_cache[target.name][2][3::4, 3::4]

    def _energy_R_Z_RKS_FD(self):
        target = Anygrad.Property.ENERGY
        self._build_fd_cache("RKS", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask, 3::4]

    def _energy_Z_R_RKS_FD(self):
        return self._energy_R_Z_RKS_FD().T

    def _homo_R_RHF_FD(self):
        target = Anygrad.Property.HOMO
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, :3].reshape(-1)

    def _homo_Z_RHF_FD(self):
        target = Anygrad.Property.HOMO
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][1].reshape(-1, 4)[:, 3].reshape(-1)

    def _homo_Z_Z_RHF_FD(self):
        target = Anygrad.Property.HOMO
        self._build_fd_cache("RHF", target)
        return self._fd_cache[target.name][2][3::4, 3::4]

    def _homo_R_R_RHF_FD(self):
        target = Anygrad.Property.HOMO
        self._build_fd_cache("RHF", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask][:, mask]

    def _homo_R_Z_RHF_FD(self):
        target = Anygrad.Property.HOMO
        self._build_fd_cache("RHF", target)
        mask = np.ones(4 * self._natm, dtype=bool)
        mask[3::4] = False
        return self._fd_cache[target.name][2][mask, 3::4]

    def _homo_Z_R_RHF_FD(self):
        return self._homo_R_Z_RHF_FD().T

    def _build_fd_cache(self, leveloftheory: str, target: "Anygrad.Property"):
        if not hasattr(self, "_fd_cache"):
            self._fd_cache = dict()
        if not target.name in self._fd_cache:
            callable = None
            is_gradient = False
            if target == Anygrad.Property.ENERGY:
                callable = lambda mf: mf.kernel()
            if target == Anygrad.Property.HOMO:

                def get_fun(calculator):
                    mocc = calculator.mo_occ > 0
                    homo_idx = calculator.mo_energy[mocc].argmax()
                    return calculator.mo_energy[mocc][homo_idx]

                callable = get_fun

                # if leveloftheory == "RHF-disabled":
                #    def get_combined_grad(calculator):
                #        Z_grad = self._homo_Z_RHF_CP(calculator)
                #        R_grad = self._homo_R_RHF_CP(calculator)
                #        combined = np.zeros((self._natm, 4))
                #        combined[:, 3] = Z_grad
                #        combined[:, :3] = R_grad.reshape(-1, 3)
                #        return combined.reshape(-1)

                #    callable = get_combined_grad
                #    is_gradient = True

            if callable is None:
                raise NotImplementedError("Don't know how to get this property.")

            if leveloftheory == "RKS":

                def build_calc(mol, xc):
                    mf = pyscf.scf.RKS(mol)
                    mf.xc = xc
                    return mf

                calc = lambda mol: build_calc(mol, self._calculator.xc)
            else:
                calc = self._calculator.__class__

            self._fd_cache[target.name] = self._finite_differences(
                self._atomspec,
                self._basis,
                calc,
                callable=callable,
                callable_is_grad=is_gradient,
            )

    class Property(enum.Enum):
        ENERGY = "energy"
        HOMO = "homo"

    class Variable(enum.Enum):
        POSITION = "R"
        NUCLEAR_CHARGE = "Z"

    class Method(enum.Enum):
        COUPLED_PERTURBED = "CP"
        FINITE_DIFFERENCES = "FD"

    def __init__(self, calculator, target: "Anygrad.Property"):
        self._calculator = calculator
        self._natm = calculator.mol.natm
        self._atomspec = calculator.mol._atom
        self._basis = calculator.mol.basis
        self._target = target

    def get(self, *args: "Anygrad.Variable", method: "Anygrad.Method" = None):
        args = tuple(Anygrad.Variable(_) for _ in args)

        # TODO: sort and auto-transpose
        if method is None:
            for method in (
                Anygrad.Method.COUPLED_PERTURBED,
                Anygrad.Method.FINITE_DIFFERENCES,
            ):
                try:
                    return self.get(*args, method=method)
                except NotImplementedError:
                    pass
            raise NotImplementedError("No method supports that derivative.")
        else:
            method = method.value
            args = tuple(_.value for _ in args)
            target = self._target.value
            leveloftheory = str(self._calculator.__class__.__name__)
            try:
                attrname = f"_{target}_{'_'.join(args)}_{leveloftheory}_{method}"
                callable = getattr(self, attrname)
            except AttributeError:
                raise NotImplementedError("This combination is not implemented.")
            return callable()

    def _finite_differences(
        self,
        atomspec: str,
        basis: str,
        baseclass: pyscf.scf.RHF,
        callable,
        delta=1e-3,
        callable_is_grad=False,
    ):
        def do_one(atomspec, basis, displacement):
            mol = pyscf.gto.M(
                atom=atomspec, basis=basis, symmetry=False, verbose=0, unit="bohr"
            )

            # update positions
            coords = mol.atom_coords(unit="Bohr")
            dx = displacement.reshape(-1, 4)[:, :3]
            dZ = displacement.reshape(-1, 4)[:, 3]
            mol.set_geom_(coords + dx, unit="Bohr")

            # update nuclear charges
            mf = baseclass(mol)
            h1 = mf.get_hcore()
            mf.max_cycle = 500

            # electronic: extend external potential
            s = 0
            for i, Z in enumerate(dZ):
                mol.set_rinv_orig_(mol.atom_coords()[i])
                s -= Z * mol.intor("int1e_rinv")

            # nuclear: difference to already included NN repulsion
            nn = 0
            for i in range(mol.natm):
                Z_i = mol.atom_charge(i) + dZ[i]

                for j in range(i + 1, mol.natm):
                    Z_j = mol.atom_charge(j) + dZ[j]

                    if i != j:
                        rij = np.linalg.norm(
                            mol.atom_coords()[i] - mol.atom_coords()[j]
                        )
                        missing = Z_i * Z_j - mol.atom_charge(j) * mol.atom_charge(i)
                        nn += missing / rij

            mf.get_hcore = lambda *args, **kwargs: h1 + s
            mf._kernel = mf.kernel
            mf.kernel = lambda *args, **kwargs: mf._kernel(*args, **kwargs) + nn
            mf.kernel()
            if not mf.converged:
                raise ValueError("SCF did not converge.")
            return callable(mf)

        ndims = 4 * pyscf.gto.M(atom=atomspec, basis=basis, symmetry=False).natm
        gradient = np.zeros(ndims)
        hessian = np.zeros((ndims, ndims))

        if callable_is_grad:
            gradient = do_one(atomspec, basis, np.zeros(ndims))
            for i in range(ndims):
                disp_i = np.zeros(ndims)
                disp_i[i] = delta
                gradient_up = do_one(atomspec, basis, disp_i)
                gradient_dn = do_one(atomspec, basis, -disp_i)
                hessian[i] = (gradient_up - gradient_dn) / (2 * delta)
            return None, gradient, hessian

        center = do_one(atomspec, basis, np.zeros(ndims))
        gradient = np.zeros(ndims)
        hessian = np.zeros((ndims, ndims))

        for i in range(ndims):
            disp_i = np.zeros(ndims)
            disp_i[i] = delta

            f_i_plus = do_one(atomspec, basis, disp_i)
            f_i_minus = do_one(atomspec, basis, -disp_i)

            gradient[i] = (f_i_plus - f_i_minus) / (2 * delta)
            hessian[i, i] = (f_i_plus - 2 * center + f_i_minus) / (delta**2)

            for j in range(i + 1, ndims):
                disp_j = np.zeros(ndims)
                disp_j[j] = delta

                f_pp = do_one(atomspec, basis, disp_i + disp_j)
                f_pm = do_one(atomspec, basis, disp_i - disp_j)
                f_mp = do_one(atomspec, basis, -disp_i + disp_j)
                f_mm = do_one(atomspec, basis, -disp_i - disp_j)  #

                hessian[i, j] = hessian[j, i] = (f_pp - f_pm - f_mp + f_mm) / (
                    4 * delta**2
                )

        return center, gradient, hessian
