import pandas as pd
import findiff
import numpy as np
import collections
import math
import itertools as it
from typing import Union

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

    def __repr__(self):
        return f"Monomial({self._prefactor}, {self._powers})"

    def prefactor(self) -> float:
        """Calculates the Taylor expansion prefactor.

        Returns
        -------
        float
            Prefactor for the summation in the Taylor expansion.
        """
        return self._prefactor / np.prod(
            [math.factorial(_) for _ in self._powers.values()]
        )

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
        return np.prod(ret)


class MultiTaylor:
    """Multi-dimensional multi-variate arbitrary order Taylor expansion from any evenly spaced finite difference stencil."""

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

    def _check_uniqueness(self, df: pd.DataFrame, terms: list[str]):
        """Checks whether the stencil is unique, i.e. whether sufficiently many columns have been filtered.

        If the raw data contains columns I1, I2, O1, O2, A and B, with all evaluated at different points, then a stencil
        over the input columns I1, I2 can be evaluated for the output columns O1 and O2 only if columns A and B are filtered
        such that only one set of values remains for each set of values in the I columns.

        Parameters
        ----------
        df : pd.DataFrame
            Stencil dataframe, possibly filtered.
        terms : list[str]
            List of columns that should be unique for a given stencil.

        Raises
        ------
        ValueError
            When there are other columns that are not unique.
        """
        copy = df.copy()

        # remove variable column
        for term in terms:
            del copy[term]

        # remove output columns: they are arbitrary
        for output in self._outputs:
            del copy[output]

        copy.drop_duplicates(inplace=True)
        if len(copy.index) > 1 and len(copy.columns) > 0:
            print(copy)
            print(copy.index)
            print(copy.columns)
            raise ValueError(f"Terms {terms} are not unique. Is a filter missing?")

    def _split_term(self, term: str, order: int) -> dict[str, int]:
        """Splits a string-based term into a dictionary based term.

        Keys are the variable names, values are the exponents.

        Parameters
        ----------
        term : str
            String-based term, e.g. A_B_C or A_A_B or A, where the letters are the variable names. All variable names must be given, only for the special case of all variable names being identical, this can be omitted.
        order : int
            The total expansion order of this term, i.e. the sum of all exponents.

        Returns
        -------
        dict[str, int]
            Explicit split term representation.

        Raises
        ------
        ValueError
            Wrong order given
        """
        parts = term.split("_")
        if len(parts) == 1:
            parts = [parts[0]] * order
        if len(parts) != order:
            raise ValueError(f"Term {term} has the wrong order.")

        return dict(collections.Counter(parts))

    def _offsets_from_df(
        self, df: pd.DataFrame, variable_columns: list[str]
    ) -> tuple[list[tuple[float]], list[float]]:
        """Transforms the available data points in the dataframe into stencil offsets.

        Parameters
        ----------
        df : pd.DataFrame
            Stencil, i.e. filtered and subselected data frame.
        variable_columns : list[str]
            List of variables to consider.

        Returns
        -------
        tuple[list[tuple[float]], list[float]]
            Offsets and spacings for the stencil.

        Raises
        ------
        ValueError
            One column is not evenly spaced.
        """
        offsets = np.zeros((len(variable_columns), len(df)), dtype=float)
        spacings = dict()

        for column in variable_columns:
            unique_values = np.sort(df[column].unique())
            spacing = np.diff(unique_values)
            if not np.allclose(spacing, spacing.mean()):
                raise ValueError(f"Variable {column} is not evenly spaced.")
            offsets[variable_columns.index(column)] = (
                df[column].values - self._center[column]
            ) / spacing.mean()
            spacings[column] = spacing.mean()

        return [tuple(_) for _ in offsets.T], [spacings[_] for _ in variable_columns]

    def _build_monomials(self, df: pd.DataFrame, term: str, order: int):
        """Builds all monomials and ensures that the stencil is applicable.

        Parameters
        ----------
        df : pd.DataFrame
            All filtered input data.
        term : str
            String-based term representation.
        order : int
            Expansion order

        Raises
        ------
        ValueError
            No stencil could be built.
        ValueError
            Not enough points in the stencil for a given order.
        """
        variable_columns = list(set(term.split("_")))
        offsets, spacings = self._offsets_from_df(df, variable_columns)
        assert len(offsets) == len(set(offsets))
        terms = self._split_term(term, order)
        partials = tuple([terms[_] for _ in variable_columns])

        try:
            stencil = findiff.stencils.Stencil(
                offsets, partials={partials: 1}, spacings=spacings
            )
        except:
            raise ValueError(f"Could not build stencil for term {term}.")
        if len(stencil.values) == 0:
            print(df)
            print(order)
            print(offsets)
            print(variable_columns)
            print(partials)
            raise ValueError(f"Not enough points for term {term}.")

        for output in self._outputs:
            weights = [stencil.values[_] if _ in stencil.values else 0 for _ in offsets]
            values = df[output].values

            self._monomials[output].append(
                Monomial(
                    prefactor=np.dot(weights, values),
                    powers=terms,
                )
            )

    def _all_terms_up_to(self, order: int) -> dict[int, list[str]]:
        """For all remaining input columns, find all possible terms entering a Taylor expansion.

        Parameters
        ----------
        order : int
            The maximum order of the expansion.

        Returns
        -------
        dict[int, list[str]]
            Order as key, list of terms as value.
        """
        terms = {}
        data_columns = [_ for _ in self._filtered.columns if _ not in self._outputs]
        data_columns = [_ for _ in data_columns if not _ in self._filter.keys()]
        for order in range(1, order + 1):
            terms[order] = [
                "_".join(_)
                for _ in it.combinations_with_replacement(data_columns, order)
            ]
        return terms

    def build_model(self, orders: Union[int, dict[int, list[str]]]):
        """Sets up the model for a specific expansion order.

        Parameters
        ----------
        orders : Union[int, dict[int, list[str]]]
            Either int, then all terms are included in the expansion up to this order. Otherwise, a dictionary with the order as key and a list of string-based terms as value.

        Raises
        ------
        NotImplementedError
            Center needs to be given in dataframe.
        ValueError
            Center is not unique.
        """
        # check center: there can be only one
        center_rows = self._dict_filter(self._dataframe, self._center)
        center_row = self._dict_filter(center_rows, self._filter)
        if len(center_row) == 0:
            raise NotImplementedError(f"Center is not in the dataframe.")
        if len(center_row) > 1:
            raise ValueError(f"Center is not unique.")

        # setup constant term
        self._monomials = {k: [Monomial(center_row.iloc[0][k])] for k in self._outputs}

        # accept integer as placeholder
        if isinstance(orders, int):
            orders = self._all_terms_up_to(orders)

        # setup other terms
        for order, terms in orders.items():
            for term in terms:
                if order == 1:
                    other_fields = {k: v for k, v in self._center.items() if k != term}
                else:
                    other_fields = {
                        k: v for k, v in self._center.items() if k not in term
                    }
                s = self._dict_filter(self._filtered, other_fields)
                self._check_uniqueness(s, self._split_term(term, order).keys())
                self._build_monomials(s, term, order)

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
