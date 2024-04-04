import pandas as pd
import findiff
import numpy as np
import math
import collections
import itertools as it

from scipy.optimize import minimize


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

        offsets = tuple(map(tuple, offsets))
        try:
            stencil = findiff.stencils.Stencil(
                offsets, partials={indices: 1}, spacings=1
            )
        except:
            raise ValueError(f"Could not build stencil for term {term}.")
        if len(stencil.values) == 0:
            raise ValueError(f"Not enough points for term {term}.")

        for output in self._outputs:
            weights = [stencil.values[_] if _ in stencil.values else 0 for _ in offsets]
            values = self._filtered[output].values

            self._monomials[output].append(
                Monomial(
                    prefactor=np.dot(weights, values),
                    powers=dict(term_counter),
                )
            )

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
        data_columns = [_ for _ in self._filtered.columns if _ not in self._outputs]
        data_columns = [_ for _ in data_columns if not _ in self._filter.keys()]
        for order in range(1, order + 1):
            for entry in it.combinations_with_replacement(data_columns, order):
                terms.append(entry)
        return tuple(terms)

    def build_model(self, orders: int):
        """Sets up the model for a specific expansion order.

        Parameters
        ----------
        orders : int
            All terms are included in the expansion up to this order.

        Raises
        ------
        NotImplementedError
            Center needs to be given in dataframe.
        ValueError
            Center is not unique.
        ValueError
            Duplicate points in the dataset.
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

        terms = self._all_terms_up_to(orders)
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
