import functools
import gzip
import pathlib
import random

import mpmath
import networkx as nx
import numpy as np
import random_graph
import scipy.special as scs
from mpmath import mp
import msgpack

from .utils import (
    AtomStoichiometry,
    Molecule,
    Q,
    SearchSpace,
    _is_pure,
    _to_pure,
)


class ApproximateCounter:
    def __init__(self, other_cachedirs: list[pathlib.Path] = None, show_progress=True):
        self._progress = show_progress
        self._seen_sequences = {}
        self._exact_cache = {}
        self._base_cache = {}
        self._pure_cache = {}
        # see SI of the paper
        self._a, self._b = 0.5758412256807119, -4.108765736350106
        self._asymptotic_a = -90.26536323333897
        self._asymptotic_b = 96.37998089390749
        self._asymptotic_c = -0.004867110462540063
        self._asymptotic_d = -0.5466985322004583
        self._asymptotic_e = 47.56524350513164
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

        return  # TODO: read cache
        cachedirs = [pathlib.Path(__file__).parent.resolve() / "cache"]
        if other_cachedirs is not None:
            cachedirs += other_cachedirs

        self._cachefiles = dict()
        kinds = "exact base pure".split()
        for kind in kinds:
            self._cachefiles[kind] = dict()

        for cachedir in cachedirs:
            for kind in kinds:
                for fn in cachedir.glob(f"space-{kind}-*.txt*"):
                    natoms = int(str(fn).split("-")[-1].split(".")[0])
                    if natoms not in self._cachefiles[kind]:
                        self._cachefiles[kind][natoms] = []
                    self._cachefiles[kind][natoms].append(fn)

        self._max_natoms_from_cache = dict()
        overall = 0
        for kind in kinds:
            self._max_natoms_from_cache[kind] = max(self._cachefiles[kind].keys())
            overall = max(overall, self._max_natoms_from_cache[kind])
        self._max_natoms_from_cache["all"] = overall
        self._estimated = []

    @staticmethod
    def read_db(fn: str) -> dict:
        """Reads the database files distributed with the package."""
        with gzip.open(fn) as fh:
            db = msgpack.load(
                fh,
                strict_map_key=False,
                use_list=False,
            )
        return db

    def estimated_in_cache(self, maxsize: int = None) -> list[str]:
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
        for natoms in range(3, self._max_natoms_from_cache["all"] + 1):
            self._fill_cache(natoms)

        if maxsize is None:
            return self._estimated
        else:
            smaller = []
            for label in self._estimated:
                lookup = self._label_to_lookup(label)
                if lookup in self._base_cache:
                    if self._base_cache[lookup] < maxsize:
                        smaller.append(label)
                    continue
                if lookup in self._pure_cache:
                    if self._pure_cache[lookup] < maxsize:
                        smaller.append(label)
                    continue
                raise NotImplementedError("Lookup failed.")
            return smaller

    def _label_to_lookup(self, label: str) -> tuple[int]:
        """Converts a canonical label into a cache lookup key.

        Parameters
        ----------
        label : str
            Canonical label ("degree.natoms_degree.natoms")

        Returns
        -------
        tuple[int]
            The corresponding cache key.
        """
        parts = label.replace(".", "_").split("_")
        return tuple(map(int, parts))

    def _parse_base_file(self, file):
        """Parse a data file containing estimated graph counts for pure and non-pure colored degree sequences.

        Parameters
        ----------
        file : str
            Filename
        """
        with gzip.open(file, "rt") as fh:
            for line in fh:
                canonical_label, length = line.split()
                lookup = self._label_to_lookup(canonical_label)
                if lookup in self._exact_cache:
                    # why estimate if we already know?
                    continue
                self._estimated.append(canonical_label)
                self._base_cache[lookup] = self._average_path_length_to_size(length)
                if _is_pure(lookup):
                    self._pure_cache[lookup] = self._base_cache[lookup]

    def _parse_pure_file(self, file: str):
        """Parse a data file containing estimated graph counts for pure colored degree sequences.

        Parameters
        ----------
        file : str
            Filename
        """
        with open(file) as fh:
            for lidx, line in enumerate(fh):
                try:
                    canonical_label, length = line.split()
                    lookup = self._label_to_lookup(canonical_label)
                    if lookup in self._exact_cache:
                        # why estimate if we already know?
                        continue
                    self._estimated.append(canonical_label)
                    self._pure_cache[lookup] = self._average_path_length_to_size(length)
                except:
                    raise ValueError(f"Cannot parse {file}, line {lidx}.")

    def _parse_exact_file(self, file: str):
        """Parse a data file containing exact graph counts for colored degree sequences.

        Parameters
        ----------
        file : str
            Filename
        """
        with open(file) as fh:
            for line in fh:
                canonical_label, count = line.split()
                canonical_label = self._label_to_lookup(canonical_label)
                self._exact_cache[canonical_label] = int(count)

    def _size_to_average_path_length(self, size: int) -> float:
        """Converts a total number of molecules to an expected average path length.

        Parameters
        ----------
        size : int
            Total count of molecules

        Returns
        -------
        float
            The corresponding average path length.
        """
        if size < 1:
            return 0
        return (np.log(float(size)) - self._b) / self._a

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

    def _fill_cache(self, natoms: int):
        """Populates the cache by parsing the file for the given number of atoms.

        Parameters
        ----------
        natoms : int
            Number of atoms for which to read the cache
        """
        functions = {
            "exact": self._parse_exact_file,
            "pure": self._parse_pure_file,
            "base": self._parse_base_file,
        }

        for kind in "exact base pure".split():  # making sure exact ones are read first
            if natoms in self._cachefiles[kind]:
                for filename in self._cachefiles[kind][natoms]:
                    functions[kind](filename)
                del self._cachefiles[kind][natoms]

        if natoms not in self._seen_sequences:
            self._seen_sequences[natoms] = {}

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
        self._fill_cache(natoms)
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
        self._fill_cache(natoms)
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
        self._fill_cache(natoms)
        sum_formulas = []
        for stoichiometry in search_space.list_cases(natoms, progress=self._progress):
            sum_formulas.append(stoichiometry.sum_formula)
        return len(set(sum_formulas))

    @staticmethod
    @functools.cache
    def factorial(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    @functools.cache
    def falling_factorial(n, k):
        result = 1
        for i in range(n, n - k, -1):
            result *= i
        return result

    @staticmethod
    @functools.cache
    def _prefactor(M: int) -> int:
        """Cached term from the paper in _count_one_asymptotically_log."""
        return ApproximateCounter.factorial(M) / (
            ApproximateCounter.factorial(M // 2) * 2 ** (M // 2)
        )

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
        float
            Log of the number estimate.
        """

        def _M(ks: list[int], r: int) -> int:
            result = 0
            for k in ks:
                result += ApproximateCounter.falling_factorial(k, r)
            return result

        M = _M(degrees, 1)
        M_2 = _M(degrees, 2)
        M_3 = _M(degrees, 3)

        prefactor = ApproximateCounter._prefactor(M)
        for k in degrees:
            prefactor /= ApproximateCounter.factorial(k)

        term1 = self._p1 * M_2 / M
        term2 = self._p2 * M_2**2 / (2 * M**2)
        term3 = M_2**4 / (4 * M**5)
        term4 = -(M_2**2 * M_3) / (2 * M**4)
        term5 = self._p3 * M_3**2 / (2 * M**3)

        paper_prefactor = prefactor
        paper_exponential = term1 + term2 + term3 + term4 + term5

        # calibration via error term
        natoms = len(degrees)

        if calibrated:
            calibration = (
                (self._asymptotic_a * natoms + self._asymptotic_b) / M
                + (self._asymptotic_c * natoms + self._asymptotic_d) * M
                + self._asymptotic_e
            )
        else:
            calibration = 0

        return np.log(float(paper_prefactor)) + paper_exponential + calibration

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
        self._fill_cache(natoms)
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

    def _pure_prediction(
        self, label: tuple[int], pure_size: int = None, log_size: float = None
    ) -> int:
        """Estimates the non-pure size via lookup of the pure average path length.

        Parameters
        ----------
        label : tuple[int]
            Case key: degree and count, alternating.
        pure_size : int
            If the size of the protomolecule list of the pure degree sequence is known, pass it here.

        Returns
        -------
        int
            Estimated number of molecules

        Raises
        ------
        KeyError
            If no pure average path length is known.
        """

        def _from_cache():
            try:
                return self._pure_cache[purespec]
            except:
                pass
            try:
                return self._exact_cache[purespec]
            except:
                pass
            try:
                return pure_size or self._base_cache[purespec]
            except:
                pass
            raise KeyError("Data missing in database")

        prefactor = self._pure_prefactor(label)
        purespec = _to_pure(label)

        if pure_size is None:
            if log_size is None:
                pure_size = _from_cache()
            else:
                pure_size = int(mpmath.exp(log_size))

        lgdu = self._size_to_average_path_length(pure_size)
        return self._average_path_length_to_size(prefactor * lgdu)

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

        # done recently?
        try:
            return self._seen_sequences[natoms][label]
        except:
            pass

        found = None
        if natoms < self._max_natoms_from_cache["all"]:
            # exact data
            if natoms <= self._max_natoms_from_cache["exact"]:
                try:
                    found = self._exact_cache[label]
                except:
                    pass
                if found is not None:
                    self._seen_sequences[natoms][label] = found
                    return found

            # average path length available
            if natoms <= self._max_natoms_from_cache["base"]:
                try:
                    found = self._base_cache[label]
                except:
                    pass
                if found is not None:
                    self._seen_sequences[natoms][label] = found
                    return found

            # reduction on pure
            if natoms <= self._max_natoms_from_cache["pure"]:
                try:
                    found = self._pure_prediction(label)
                except KeyError:
                    pass
                if found is not None:
                    self._seen_sequences[natoms][label] = found
                    return found

        # only use asymptotic scaling relations if the number of atoms is large enough

        if natoms < self._minimum_natoms_for_asymptotics:
            raise ValueError(
                f"""The pre-computed database does not cover this stoichiometry: {label}.
                
                You may either compute it yourself using estimate_edit_tree_average_path_length()
                (see maintenance/space_cache.py) or contact vonrudorff@uni-kassel.de, 
                so we can distribute the extended cache in a new version of this library, 
                as building the cache is a time-consuming process."""
            )
        if cached_degree_sequence:
            log_asymptotic_size = self._cached_log_asymptotic_size
        else:
            degrees = sum([[v] * c for v, c in zip(label[::2], label[1::2])], [])
            log_asymptotic_size = self._count_one_asymptotically_log(tuple(degrees))
            self._cached_log_asymptotic_size = log_asymptotic_size

        if _is_pure(label):
            found = int(mpmath.exp(log_asymptotic_size))
        else:
            found = self._pure_prediction(label, log_size=log_asymptotic_size)
        self._seen_sequences[natoms][label] = found
        return found

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

    @staticmethod
    @staticmethod
    def _weighted_choose(items: list, weights: list[int]):
        """Weighted random selection with support for arbitrarily large integer weights.

        Parameters
        ----------
        items : list
            Items to choose from.
        weights : list[int]
            Integer weights for each item.

        Returns
        -------
        any
            One item.
        """
        total = sum(weights)
        choice = random.randint(0, total)
        for item, weight in zip(items, weights):
            choice -= weight
            if choice <= 0:
                return item

    def _sum_formula_database(
        self, search_space: SearchSpace, natoms: int, selection: Q = None
    ):
        """Builds a size database of the sizes of all sum formulas in randomized order.

        Parameters
        ----------
        search_space : SearchSpace
            The search space.
        natoms : int
            The number of atoms to cover.
        selection : Q, optional
            A selection via query string, by default None

        Returns
        -------
        tuple[list[str], list[int], dict[str, list[utils.AtomStoichiometry]]]
            Sum formulas in random order, their molecule count and all stoichiometries for all sum formulas.
        """
        sum_formula_size = {}
        stoichiometries = {}
        for stoichiometry in search_space.list_cases(natoms, progress=self._progress):
            if selection is not None and not selection.selected_stoichiometry(
                stoichiometry
            ):
                continue
            sum_formula = stoichiometry.sum_formula
            magnitude = self.count_one(stoichiometry, natoms)
            if sum_formula not in stoichiometries:
                stoichiometries[sum_formula] = [stoichiometry]
                sum_formula_size[sum_formula] = 0
            sum_formula_size[sum_formula] += magnitude

        random_order = list(sum_formula_size.keys())
        random.shuffle(random_order)
        sizes = [sum_formula_size[_] for _ in random_order]

        return random_order, sizes, stoichiometries

    def random_sample(
        self,
        search_space: SearchSpace,
        natoms: int,
        nmols: int,
        selection: Q = None,
    ) -> list[Molecule]:
        """Builds a fixed size random sample from a search space for a given number of atoms.

        Parameters
        ----------
        search_space : SearchSpace
            The total search space.
        natoms : int
            The total number of atoms of the chosen molecules.
        nmols : int
            Number of random molecules to be drawn.
        selection : Q, optional
            Selecting subsets via query string, by default None

        Returns
        -------
        list[Molecule]
            Random molecules.

        Raises
        ------
        ValueError
            If selection is empty.
        """
        random_order, sizes, stoichiometries = self._sum_formula_database(
            search_space, natoms, selection
        )
        if len(random_order) == 0 or sum(sizes) == 0:
            raise ValueError("Search space and selection yield no feasible molecule.")

        # sample molecules
        molecules = []
        while len(molecules) < nmols:
            sum_formula = ApproximateCounter._weighted_choose(random_order, sizes)
            ss = stoichiometries[sum_formula]
            ws = [self.count_one(_, natoms) for _ in ss]
            stoichiometry = ApproximateCounter._weighted_choose(ss, ws)
            spec = stoichiometry.canonical_label

            # try a few times to find a concrete graph
            # needs to stop eventually, since there might be stoichiometries
            # for which the estimate is non-zero but which are actually
            # infeasible
            tries_selection = 0
            while True and tries_selection < 1e4:
                graph = ApproximateCounter.sample_connected(spec)

                # correct for isomorphisms which skew MC sampling
                degrees, _ = ApproximateCounter.spec_to_sequence(spec)
                underlying = nx.Graph(graph)
                nx.set_node_attributes(underlying, dict(enumerate(degrees)), "element")
                underlying_isomorphisms = len(
                    list(
                        nx.vf2pp_all_isomorphisms(
                            underlying, underlying, node_label="element"
                        )
                    )
                )
                graph_iso = len(
                    list(nx.vf2pp_all_isomorphisms(graph, graph, node_label="element"))
                )

                weight = graph_iso / underlying_isomorphisms
                if random.random() < weight:
                    sorted_nodes = sorted(
                        graph.nodes, key=lambda x: graph.nodes[x]["element"]
                    )
                    labels = stoichiometry.canonical_element_sequence
                    graph = nx.relabel_nodes(
                        graph, dict(zip(sorted_nodes, range(len(labels)))), copy=True
                    )
                    mol = Molecule(labels, list(graph.edges(data=False)))
                    if selection and not selection.selected_molecule(mol):
                        tries_selection += 1
                        continue
                    molecules.append(mol)
                    break

        return molecules

    def score_database(
        self,
        search_space: SearchSpace,
        natoms: int,
        sum_formulas: dict[str, int],
        selection: Q = None,
    ) -> float:
        """Implements the Kolmogorov-Smirnov statistic comparing the distribution of the database to the expected distribution.

        The best score is 0, the worst is 1.

        This does not test the distribution of molecules within a given sum formula."""

        random_order, sizes, _ = self._sum_formula_database(
            search_space, natoms, selection=selection
        )

        xs = np.cumsum(sizes)
        ys_expected = xs / xs[-1]
        ys = []
        for sum_formula in random_order:
            if sum_formula in sum_formulas:
                ys.append(sum_formulas[sum_formula])
            else:
                ys.append(0)
        ys = np.cumsum(ys)
        ys = ys / ys[-1]

        return max(abs(ys - ys_expected))

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

        self._fill_cache(natoms)
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
                if natoms <= self._max_natoms_from_cache["exact"]:
                    if label in self._exact_cache:
                        continue

                if natoms <= self._max_natoms_from_cache["base"]:
                    if label in self._base_cache:
                        continue

                # reduction on pure
                if natoms <= self._max_natoms_from_cache["pure"]:
                    if pure_only:
                        purespec = label
                    else:
                        purespec = _to_pure(label)

                    if purespec in self._pure_cache or purespec in self._base_cache:
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
                if (
                    label not in self._exact_cache
                    and label not in self._base_cache
                    and label not in self._pure_cache
                ):
                    missing.append(label)

        return list(set(missing))
