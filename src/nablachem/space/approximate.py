import functools
import pathlib
from typing import Generator
import networkx as nx
import numpy as np
import random_graph
import scipy.special as scs
from mpmath import mp
from .utils import (
    AtomStoichiometry,
    Molecule,
    Q,
    SearchSpace,
    _is_pure,
    _to_pure,
    _read_db,
    falling_factorial,
    factorial,
)


class ApproximateCounter:
    def __init__(self, show_progress=True):
        self._progress = show_progress
        self._seen_sequences = {}
        self._exact_cache = {}
        self._base_cache = {}
        self._pure_cache = {}
        # coefficients from the paper, obtained with maintenance/space_fit.py
        self._a, self._b = 1.22066271, -0.72953204
        self._asymptotic_a = 0.75617982
        self._asymptotic_b = -14.40107699
        self._minimum_natoms_for_asymptotics = 20

        # cache mpf objects for performance
        one = mp.mpf("1")
        three = mp.mpf("3")
        third = one / three
        half = one / mp.mpf("2")

        y1 = mp.mpf("0")  # no loops allowed
        x2 = mp.mpf("1")  # double bonds allowed
        x3 = mp.mpf("1")  # triple bonds allowed

        self._p1 = y1 - half
        self._p2 = x2 - half
        self._p3 = x3 - x2 + third

        # read cache
        cachedir = pathlib.Path(__file__).parent.resolve() / ".." / "cache"
        self._exact_db = _read_db(cachedir / "space-exact.msgpack.gz")
        self._approx_db = _read_db(cachedir / "space-approx.msgpack.gz")

        # transform approximate path length into integer counts
        keys_to_delete = []
        updates = {}
        for key, value in list(self._approx_db.items()):
            canonical_key = self._canonical_label(key)
            if canonical_key != key:
                keys_to_delete.append(key)
            updates[canonical_key] = self._average_path_length_to_size(value)
        for key in keys_to_delete:
            del self._approx_db[key]
        self._approx_db.update(updates)

        # canonicalize exact key
        updates = {}
        for key, value in self._exact_db.items():
            canonical_key = self._canonical_label(key)
            updates[canonical_key] = value
        self._exact_db.update(updates)

    def estimated_in_cache(self, maxsize: int = None) -> Generator[str, None, None]:
        """Builds a list of all those degree sequences that are estimated only.

        Parameters
        ----------
        maxsize : int, optional
            Cap of the estimated size of the number of graphs with that degree sequence, by default None

        Returns
        -------
        list[str]
            List of canonical labels
        """
        for key, value in self._approx_db.items():
            if key in self._exact_db:
                continue
            if maxsize is not None and value > maxsize:
                continue
            yield "_".join(
                [f"{degree}.{count}" for degree, count in zip(key[::2], key[1::2])]
            )

    def _average_path_length_to_size(self, length: float) -> int:
        """Converts the average path length from the database to a molecule count.

        Parameters
        ----------
        length : float
            The average path length.

        Returns
        -------
        int
            Total number of molecules.
        """
        return max(int(np.exp(self._a * float(length) + self._b)), 0)

    def _size_to_average_path_length(self, size: int) -> float:
        return (max(np.log(float(size)), 1) - self._b) / self._a

    def count(self, search_space: SearchSpace, natoms: int, selection: Q = None) -> int:
        """Counts the total number of molecules in a search space.

        Parameters
        ----------
        search_space : SearchSpace
            The search space.
        natoms : int
            The number of atoms to restrict to.
        selection : Q, optional
            A subselection based on a query string, by default None

        Returns
        -------
        int
            Total count of molecules in this search space.
        """
        total = 0

        if selection:
            for stoichiometry in search_space.list_cases(
                natoms, progress=self._progress
            ):
                if not selection.selected_stoichiometry(stoichiometry):
                    continue
                total += self.count_one(stoichiometry, natoms, validated=True)
        else:
            cached_degree_sequence = False
            for case in search_space.list_cases_bare(natoms, progress=self._progress):
                if case is None:
                    cached_degree_sequence = False
                    continue

                components = [[valence, count] for _, valence, count in case]
                total += self.count_one_bare(
                    tuple(sum(components, [])),
                    natoms,
                    cached_degree_sequence,
                    validated=True,
                )
                cached_degree_sequence = True
        return total

    def count_cases(self, search_space: SearchSpace, natoms: int) -> int:
        """Counts the total number of stoichiometries in a search space.

        Note that different stoichiometries could yield the same sum formula.

        Note that this only returns cases where all valences are saturated.

        Parameters
        ----------
        search_space : SearchSpace
            The search space.
        natoms : int
            The number of atoms for which to count.

        Returns
        -------
        int
            Total number of stoichiometries.
        """
        total = 0
        for case in search_space.list_cases_bare(natoms, progress=self._progress):
            if case is not None:
                total += 1
        return total

    def count_sum_formulas(self, search_space: SearchSpace, natoms: int) -> int:
        """Counts the total number of sum formulas in a search space.

        Note that this only returns cases where all valences are saturated.

        Parameters
        ----------
        search_space : SearchSpace
            The search space.
        natoms : int
            The number of atoms for which to count.

        Returns
        -------
        int
            Total number of sum formulas.
        """
        sum_formulas = []
        for stoichiometry in search_space.list_cases(natoms, progress=self._progress):
            sum_formulas.append(stoichiometry.sum_formula)
        return len(set(sum_formulas))

    @staticmethod
    @functools.cache
    def _prefactor(M: int) -> int:
        """Cached term from the paper in _count_one_asymptotically_log."""
        return factorial(M) / (factorial(M // 2) * 2 ** (M // 2))

    @functools.cache
    def _count_one_asymptotically_log(
        self, degrees: tuple[int], calibrated: bool = True
    ):
        """Estimate the total number of molecules of a given degree sequence from the asymptotic limit.

        Follows "Asymptotic Enumeration of Sparse Multigraphs with Given Degrees"
        C Greenhill, B McKay, SIAM J Discrete Math. 10.1137/130913419, Theorem 1.1.

        The calibration has been added by nablachem to determine the prefactors of the asymptotic
        formulas.

        Parameters
        ----------
        degrees : tuple[int]
            Degree sequence
        calibrated : bool, optional
            Whether to apply the calibration, by default True

        Returns
        -------
        int, float
            Number estimate if calibrated or log estimate if not calibrated.
        """

        def _M(ks: list[int], r: int) -> int:
            result = 0
            for k in ks:
                result += falling_factorial(k, r)
            return result

        M = _M(degrees, 1)
        M_2 = _M(degrees, 2)
        M_3 = _M(degrees, 3)

        prefactor = ApproximateCounter._prefactor(M)
        for k in degrees:
            prefactor /= factorial(k)

        term1 = self._p1 * M_2 / M
        term2 = self._p2 * M_2**2 / (2 * M**2)
        term3 = M_2**4 / (4 * M**5)
        term4 = -(M_2**2 * M_3) / (2 * M**4)
        term5 = self._p3 * M_3**2 / (2 * M**3)

        paper_prefactor = prefactor
        paper_exponential = term1 + term2 + term3 + term4 + term5
        logG = np.log(float(paper_prefactor)) + paper_exponential
        if not calibrated:
            return logG

        lg = self._asymptotic_a * logG + self._asymptotic_b
        return self._average_path_length_to_size(lg)

    def count_one(
        self, stoichiometry: AtomStoichiometry, natoms: int, validated: bool = False
    ) -> int:
        """Counts the total number of molecules in a given stoichiometry.

        The redundant specification of the number of atoms is a performance tweak.

        Parameters
        ----------
        stoichiometry : AtomStoichiometry
            The stoichiometry to count.
        natoms : int
            Number of atoms in that stoichiometry.
        validated : bool, optional
            Whether the given degree sequence needs to be checked for feasibility. When in doubt, True

        Returns
        -------
        int
            Total count of molecules.
        """
        return self.count_one_bare(
            stoichiometry.canonical_tuple, natoms, validated=validated
        )

    @functools.cache
    def _cached_permutation_factor_log(self, groups: tuple[int]) -> float:
        """Calculates the natural logarithm of of permutations in groups.

        In estimating the size of non-pure degree sequences (e.g. CF2H2) from their pure counterpart
        (here, CH4), the total count of permutations is required where the non-pure groups of same
        valency are assigned to the total number of atoms of this valency.

        Parameters
        ----------
        groups : tuple[int]
            Counts of atoms of same valency.

        Returns
        -------
        float
            Log of the number of permutations.
        """
        score = 1
        remaining = sum(groups)
        for count in groups:
            score *= int(scs.binom(remaining, count))
            remaining -= count
        return np.log(score)

    def _pure_prefactor(self, label: tuple[int]) -> float:
        M = 0
        logscore = 0

        last_d = label[0]
        counts = []
        for i in range(len(label) // 2):
            degree, count = label[i * 2 : i * 2 + 2]
            if degree != last_d:
                logscore += self._cached_permutation_factor_log(tuple(counts))
                counts = sum(counts)
                M += last_d * counts
                last_d = degree
                counts = []
            counts.append(count)
        logscore += self._cached_permutation_factor_log(tuple(counts))
        counts = sum(counts)
        M += last_d * counts

        return logscore / M + 1

    def _canonical_label(self, label: tuple[int]) -> tuple[int]:
        """Converts a degree sequence to a canonical label.

        Parameters
        ----------
        label : tuple[int]
            Degree sequence in groups ((degree, natoms), (degree, natoms))

        Returns
        -------
        tuple[int]
            Canonical label of the degree sequence.
        """
        valences, counts = label[::2], label[1::2]
        pairs = sorted(zip(valences, counts))
        return tuple(sum([list(_) for _ in pairs], []))

    def count_one_bare(
        self,
        label: tuple[int],
        natoms: int,
        cached_degree_sequence: bool = False,
        validated: bool = False,
    ) -> int:
        """Counts the number of molecules of a given colored degree sequence.

        The last two arguments are performance tweaks and not strictly necessary.

        Parameters
        ----------
        label : tuple[int]
            Degree sequence in groups ((degree, natoms), (degree, natoms))
        natoms : int
            The total number of atoms.
        cached_degree_sequence : bool, optional
            Whether this case has the same pure degree sequence of the previous call, by default False
        validated : bool, optional
            Whether the given degree sequence needs to be checked for feasibility. When in doubt, True

        Returns
        -------
        int
            Total count.
        """
        # validated: do not check again
        if not validated:
            total = 0
            count = 0
            maxvalence = 0
            for valence, valenceatoms in zip(label[::2], label[1::2]):
                if valenceatoms == 0:
                    return 0
                total += valenceatoms * valence
                maxvalence = max(maxvalence, valence)
                count += valenceatoms
            if total % 2 != 0:
                # odd number of bonds
                return 0
            if maxvalence * 2 > total:
                # no self-loops allowed
                return 0
            dbe = int(total / 2) - (count - 1)
            if dbe < 0:
                return 0

        # canonical lookup
        label = self._canonical_label(label)
        if label in self._exact_db:
            return self._exact_db[label]
        if label in self._approx_db:
            return self._approx_db[label]

        # try to find pure degree sequence
        pure_label = _to_pure(label)
        prefactor = self._pure_prefactor(label)
        pure_counts = None
        if pure_label in self._exact_db:
            pure_counts = self._exact_db[pure_label]
        if pure_label in self._approx_db:
            pure_counts = self._approx_db[pure_label]
        if pure_counts:
            lg_pure = self._size_to_average_path_length(pure_counts)
            lg_nonpure = lg_pure * prefactor
            return self._average_path_length_to_size(lg_nonpure)

        if natoms < self._minimum_natoms_for_asymptotics:
            raise ValueError(
                f"""The pre-computed database does not cover this stoichiometry: {label}.
                
                You may either compute it yourself using estimate_edit_tree_average_path_length()
                (see maintenance/space_cache.py) or contact vonrudorff@uni-kassel.de, 
                so we can distribute the extended cache in a new version of this library, 
                as building the cache is a time-consuming process."""
            )
        degrees = sum([[v] * c for v, c in zip(label[::2], label[1::2])], [])
        terminals = sum(1 for d in degrees if d == 1)
        counts = self._count_one_asymptotically_log(tuple(degrees)) // factorial(
            terminals
        )

        if _is_pure(label):
            return counts
        else:
            lg = self._size_to_average_path_length(counts)
            lgnonpure = prefactor * lg
            return self._average_path_length_to_size(lgnonpure)

    @staticmethod
    def sample_connected(spec: str) -> nx.MultiGraph:
        """Find a random connected multigraph of a given degree sequence.

        Parameters
        ----------
        spec : str
            Canonical label from which the degree sequence is extracted.

        Returns
        -------
        nx.MultiGraph
            The resulting graph
        """
        degrees, elements = ApproximateCounter.spec_to_sequence(spec)
        while True:
            edges = random_graph.sample_multi_hypergraph(
                degrees, [2] * (sum(degrees) // 2), n_iter=1000
            )
            G = nx.MultiGraph(edges)
            if nx.connected.is_connected(G):
                nx.set_node_attributes(
                    G, dict(zip(range(len(degrees)), elements)), "element"
                )
                return G

    @staticmethod
    def spec_to_sequence(spec: str) -> tuple[list[int], list[str]]:
        """Converts a canonical label to a degree sequence and pseudoelement labels.

        Parameters
        ----------
        spec : str
            Canonical label of the pseudomolecule ("degree.natoms_degree.natoms").

        Returns
        -------
        tuple[list[int], list[str]]
            Degree sequence and artificial element labels.
        """
        degrees = []
        elements = []
        letter = "a"
        for part in spec.split("_"):
            degree, count = part.split(".")
            degrees += [int(degree)] * int(count)
            elements += [f"{letter}{degree}"] * int(count)
            letter = chr(ord(letter) + 1)
        return degrees, elements

    def missing_parameters(
        self,
        search_space: SearchSpace,
        natoms: int,
        pure_only: bool,
        selection: Q = None,
    ):
        """Returns the colored degree sequences for which no parameters are available in the database.

        Parameters
        ----------
        search_space : SearchSpace
            The search space.
        natoms : int
            Number of atoms to check for.
        pure_only : bool
            Whether only pure degree sequences should be precomputed for this space.
        selection : Q, optional
            Subselection by query string, by default None

        Returns
        -------
        list[tuple[int]]
            The colored degree sequences for which there are no parameters.
        """
        # no parameters needed for asymptotics
        if natoms > self._minimum_natoms_for_asymptotics:
            return []

        missing = []

        if selection is not None:
            for case in search_space.list_cases_bare(natoms, progress=self._progress):
                if case is None:
                    continue

                label = tuple(sum([[valence, count] for _, valence, count in case], []))
                if pure_only and not _is_pure(label):
                    continue

                if selection is not None and not selection.selected_stoichiometry(
                    sum([[element] * count for element, _, count in case], [])
                ):
                    continue

                # check in cache
                label = self._canonical_label(label)
                if label in self._approx_db:
                    continue
                if label in self._exact_db:
                    continue

                purespec = self._canonical_label(_to_pure(label))
                if purespec in self._approx_db:
                    continue
                if purespec in self._exact_db:
                    continue
                missing.append(label)
        else:
            for case in search_space.list_cases_bare(
                natoms,
                degree_sequences_only=True,
                pure_sequences_only=pure_only,
                progress=self._progress,
            ):
                if case is None:
                    continue

                label = tuple(sum([[valence, count] for valence, count in case], []))
                label = self._canonical_label(label)
                if label not in self._exact_db and label not in self._approx_db:
                    missing.append(label)

        return list(set(missing))
