import subprocess
from collections.abc import Iterator, Callable
import functools
import re
import pathlib
import networkx as nx
import itertools as it
import random_graph
import random
import scipy.optimize as sco
import numpy as np
import collections
import math
import operator
from scipy._lib._util import check_random_state
from scipy.optimize._optimize import _check_unknown_options
from scipy.optimize import OptimizeResult
import pyparsing
import operator
import functools
import gzip
from mpmath import mp
import mpmath
from .utils import *


class SearchSpace:
    def __init__(self, elements: str = None):
        self._atom_types = []
        if elements is not None:
            for elementspec in elements.strip().split():
                element, valences = elementspec.split(":")
                valences = [int(_) for _ in valences.split(",")]
                self.add_element(Element(element, valences))

    def add_element(self, element: Element):
        self._atom_types += element.atom_types

    def list_cases_bare(
        self,
        natoms: int,
        degree_sequences_only: bool = False,
        pure_sequences_only: bool = False,
    ) -> Iterator[tuple[str, int, int]] | Iterator[tuple[int, int]]:
        """Lists all possible stoichiometries for a given number of atoms.

        If degree_sequences_only is set to True, only unique degree sequences are
        returned, i.e. the element names are not considered.

        Optimized for performance, so yields tuples. Use list_cases() for a more
        user-friendly interface.

        Parameters
        ----------
        natoms : int
            Number of atoms in the molecule.
        degree_sequences_only : bool, optional
            Flag to switch to degree sequence enumeration, by default False
        pure_sequences_only : bool, optional
            Skips sequences where atoms of one valence belong to more than one element label. Implies degree_sequences_only.

        Yields
        ------
        Iterator[tuple[str, int, int]] | Iterator[tuple[int, int]]
            Either tuples of (element, valence, count) or (valence, count). Guaranteed to be sorted by (valence, count).
        """
        valences = sorted(set([_.valence for _ in self._atom_types]))
        valence_elements = [self.get_elements_from_valence(v) for v in valences]

        if pure_sequences_only:
            degree_sequences_only = True
            valence_elements = [_[:1] for _ in valence_elements]

        @functools.lru_cache(maxsize=None)
        def get_group(nvalenceatoms: int, valenceidx: int):
            kinds = valence_elements[valenceidx]
            valence = valences[valenceidx]
            group = []
            seen = []
            for inner_partition in integer_partition(nvalenceatoms, len(kinds)):
                if degree_sequences_only:
                    if sorted(inner_partition) in seen:
                        continue
                    seen.append(sorted(inner_partition))

                case = []
                for element, count in zip(kinds, inner_partition):
                    if count > 0:
                        if degree_sequences_only:
                            case.append((valence, count))
                        else:
                            case.append((element, valence, count))
                case.sort(key=lambda x: x[-1])
                group.append(case)
            return group

        for outer_partition in integer_partition(natoms, len(valences)):
            total = 0
            count = 0
            maxvalence = 0
            for valenceatoms, valence in zip(outer_partition, valences):
                if valenceatoms == 0:
                    continue
                total += valenceatoms * valence
                maxvalence = max(maxvalence, valence)
                count += valenceatoms
            if total % 2 != 0:
                # odd number of bonds
                continue
            if maxvalence * 2 > total:
                # no self-loops allowed
                continue
            dbe = int(total / 2) - (count - 1)
            if dbe < 0:
                continue

            groups = []
            for valenceidx, nvalenceatoms in enumerate(outer_partition):
                if nvalenceatoms == 0:
                    continue

                groups.append(
                    get_group(
                        nvalenceatoms,
                        valenceidx,
                    )
                )

            for case in it.product(*groups):
                yield sum(case, [])

    def list_cases(self, natoms: int) -> Iterator[AtomStoichiometry]:
        for case in self.list_cases_bare(natoms):
            stoichiometry = AtomStoichiometry()
            for element, valence, count in case:
                stoichiometry.extend(AtomType(element, valence), count)
            yield stoichiometry

    @property
    def max_valence(self):
        return max([_.valence for _ in self._atom_types])

    def get_elements_from_valence(self, valence):
        return [e.label for e in self._atom_types if e.valence == valence]

    @staticmethod
    def covered_search_space():
        s = SearchSpace()
        s.add_element(Element("C", [4]))
        s.add_element(Element("N", [3, 5]))
        s.add_element(Element("O", [2]))
        s.add_element(Element("F", [1]))
        s.add_element(Element("H", [1]))
        s.add_element(Element("Cl", [1]))
        s.add_element(Element("Br", [1]))
        s.add_element(Element("I", [1]))
        s.add_element(Element("P", [3, 5]))
        s.add_element(Element("S", [2, 4, 6]))
        s.add_element(Element("Si", [4]))
        return s


class Q:
    def __init__(self, query_string: str):
        self._parsed = Q._parse_query(query_string)

    @staticmethod
    def _parse_query(query_string: str):
        if query_string.strip() == "":
            return []
        identifier = pyparsing.Combine(
            pyparsing.Word(pyparsing.alphas + "#", pyparsing.alphanums)
        )
        number = pyparsing.Word(pyparsing.nums)
        operand = identifier | number

        comparison_op = pyparsing.oneOf("< > = <= >= !=")
        and_ = pyparsing.oneOf("and &", caseless=True)
        or_ = pyparsing.oneOf("or |", caseless=True)
        not_ = pyparsing.oneOf("not no !", caseless=True)

        parser = pyparsing.infixNotation(
            operand,
            [
                (not_, 1, pyparsing.opAssoc.RIGHT),
                (comparison_op, 2, pyparsing.opAssoc.LEFT),
                (and_, 2, pyparsing.opAssoc.LEFT),
                (or_, 2, pyparsing.opAssoc.LEFT),
            ],
        )
        return parser.parseString(query_string, parseAll=True).as_list()

    def selected_stoichiometry(self, stoichiometry: AtomStoichiometry) -> bool:
        element_counts = collections.Counter(stoichiometry.canonical_element_sequence)
        element_counts["#"] = element_counts.total()

        operators = {
            ">": operator.gt,
            "<": operator.lt,
            "=": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            "<=": operator.le,
            "and": operator.and_,
            "or": operator.or_,
            "&": operator.and_,
            "|": operator.or_,
            "not": operator.not_,
            "no": operator.not_,
            "!": operator.not_,
        }

        def evaluate(parsed):
            print(parsed)
            if parsed == []:
                return True

            if isinstance(parsed, str):
                return element_counts[parsed] > 0

            if len(parsed) == 1:
                return evaluate(parsed[0])

            if len(parsed) == 2:
                op, rhs = parsed
                return operators[op](evaluate(rhs))

            lhs, op, rhs = parsed

            if isinstance(lhs, str) and isinstance(rhs, str) and isinstance(op, str):
                try:
                    lhs = int(lhs)
                except ValueError:
                    lhs = element_counts[lhs]
                try:
                    rhs = int(rhs)
                except ValueError:
                    rhs = element_counts[rhs]
                return operators[op](lhs, rhs)
            else:
                return operators[op](evaluate(lhs), evaluate(rhs))

        return evaluate(self._parsed)


class ExactCounter:
    def __init__(self, binary):
        self._binary = binary
        self._cache = {}

    def _build_cli_arguments(
        self, stoichiometry: AtomStoichiometry, count_only: bool
    ) -> tuple[str, dict[str, str]]:
        max_valence = stoichiometry.largest_valence
        args = [f"-c{max_valence}", f"-d{max_valence}"]
        sf = ""

        letter = "a"
        elements = {}
        for atom_type, natoms in stoichiometry.components.items():
            valence = atom_type.valence
            elements[f"A{letter}"] = atom_type.label
            args.append(f"-EA{letter}{valence}{valence}")
            sf += f"A{letter}{natoms}"
            letter = chr(ord(letter) + 1)

        if count_only:
            extra = "u"
        else:
            extra = "A"

        return " ".join(args) + f" -{extra} " + sf, elements

    def count(self, search_space: SearchSpace, natoms: int) -> int:
        total = 0
        for stoichiometry in search_space.list_cases(natoms):
            total += self.count_one(stoichiometry)
        return total

    def _run(self, args: str, keep_stderr: bool = True) -> str:
        cmd = f"{self._binary} {args}"
        if keep_stderr:
            stderr = subprocess.STDOUT
        else:
            stderr = subprocess.DEVNULL
        stdout = subprocess.check_output(cmd, shell=True, stderr=stderr)
        return stdout.decode("utf-8")

    def count_one(self, stoichiometry: AtomStoichiometry):
        # cached?
        label = stoichiometry.canonical_label
        if label in self._cache:
            return self._cache[label]

        # run surge
        args, _ = self._build_cli_arguments(stoichiometry, count_only=True)
        stdout = self._run(args)
        try:
            match = [_ for _ in stdout.split("\n") if ">Z generated" in _][0]
        except:
            print(args)
            print(stdout)
            raise NotImplementedError()
        count = int(match.split()[-4])
        self._cache[label] = count
        return count

    def save(self, path):
        with open(path, "w") as f:
            for label, count in self._cache.items():
                f.write(f"{label} {count}\n")

    def load(self, path):
        with open(path) as f:
            for line in f:
                label, count = line.split()
                self._cache[label] = int(count)

    @staticmethod
    def _split_element_label(label: str) -> list[str]:
        """Splits a surge output spec of a stoichiometry into a list of elements.

        Parameters
        ----------
        label : str
            Stoichiometry spec, e.g. "Aa2Ab2Ac". Only works for element labels generated in this class.

        Returns
        -------
        list[str]
            Repeated list of element labels in order, e.g. ["Aa", "Aa", "Ab", "Ab", "Ac"].
        """
        elements = []
        for element, repeat in re.findall("(A[a-z])([0-9]*)", label):
            elements += [element] * int(repeat or 1)
        return elements

    def list_one(self, stoichiometry: AtomStoichiometry) -> list[Molecule]:
        args, lookup = self._build_cli_arguments(stoichiometry, count_only=False)

        stdout = self._run(args, keep_stderr=False)

        # node labels
        node_labels = []
        for component in stoichiometry.components.items():
            node_labels += [component[0].label] * component[1]

        bondtypes = {"-": 1, "=": 2, "#": 3}
        molecules = []
        for line in stdout.split("\n"):
            if len(line.strip()) == 0:
                continue

            # parse elements
            elements = [
                lookup[_] for _ in ExactCounter._split_element_label(line.split()[2])
            ]

            # parse edges
            edges = []
            bonds = line.strip().split()[3:]

            for bond in bonds:
                idx1, bondtype, idx2 = re.match(
                    "^([0-9]+)([^0-9])([0-9]+)$", bond
                ).groups()
                order = bondtypes[bondtype]
                edges.append([int(idx1), int(idx2), order])

            molecules.append(Molecule(elements, edges))

        return molecules

    def list(self, search_space: SearchSpace, natoms: int) -> list[Molecule]:
        molecules = []

        for stoichiometry in search_space.list_cases(natoms):
            molecules += self.list_one(stoichiometry)
        return molecules


class ApproximateCounter:
    def __init__(self, other_cachedirs: list[pathlib.Path] = None):
        self._exact_cache = {}
        self._base_cache = {}
        self._pure_cache = {}

        cachedirs = [pathlib.Path(__file__).parent.resolve() / "cache"]
        if other_cachedirs is not None:
            cachedirs += other_cachedirs

        def label_to_lookup(label):
            parts = label.replace(".", "_").split("_")
            return tuple(map(int, parts))

        for cachedir in cachedirs:
            for file in cachedir.glob("space-exact-*.txt"):
                with open(file) as fh:
                    for line in fh:
                        canonical_label, count = line.split()
                        canonical_label = label_to_lookup(canonical_label)
                        self._exact_cache[canonical_label] = int(count)

            a, b = 0.5758412256807119, -4.108765736350106
            for file in cachedir.glob("space-base-*.txt.gz"):
                with gzip.open(file, "rt") as fh:
                    for line in fh:
                        canonical_label, count = line.split()
                        canonical_label = label_to_lookup(canonical_label)
                        self._base_cache[canonical_label] = int(np.exp(a * float(count) + b))

            for file in cachedir.glob("space-pure-*.txt"):
                with open(file) as fh:
                    for lidx, line in enumerate(fh):
                        try:
                            canonical_label, count = line.split()
                            canonical_label = label_to_lookup(canonical_label)
                            self._pure_cache[canonical_label] = float(count)  # scale
                        except:
                            raise ValueError(f"Cannot parse {file}, line {lidx}.")

    def get_cache(self, natoms: int):
        def key_to_natoms(key):
            return sum([int(_.split(".")[1]) for _ in key.split("_")])

        ret = {}
        for label, cache in (
            ("exact", self._exact_cache),
            ("base", self._base_cache),
            ("pure", self._pure_cache),
        ):
            ret[label] = {k: v for k, v in cache.items() if key_to_natoms(k) == natoms}
        return ret

    def count(self, search_space: SearchSpace, natoms: int, selection: Q = None) -> int:
        total = 0

        if selection:
            for stoichiometry in search_space.list_cases(natoms):
                if not selection.selected_stoichiometry(stoichiometry):
                    continue
                total += self.count_one(stoichiometry)
        else:
            for case in search_space.list_cases_bare(natoms):
                components = [[valence, count] for _, valence, count in case]
                total += self.count_one_bare(tuple(sum(components, [])))
        return total

    def _count_one_asymptotically(self, degrees: list[int]):
        """Follows "Asymptotic Enumeration of Sparse Multigraphs with Given Degrees"
        C Greenhill, B McKay, SIAM J Discrete Math. 10.1137/130913419."""

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

        one = mp.mpf("1")
        three = mp.mpf("3")
        third = one / three
        half = one / mp.mpf("2")

        y1 = mp.mpf("0")  # no loops allowed
        x2 = mp.mpf("1")  # double bonds allowed
        x3 = mp.mpf("1")  # triple bonds allowed

        def _M(ks: list[int], r: int) -> int:
            result = 0
            for k in ks:
                result += falling_factorial(k, r)
            return result

        M = _M(degrees, 1)
        M_2 = _M(degrees, 2)
        M_3 = _M(degrees, 3)

        prefactor = factorial(M) / (factorial(M // 2) * 2 ** (M // 2))
        for k in degrees:
            prefactor /= factorial(k)

        term1 = (y1 - half) * M_2 / M
        term2 = (x2 - half) * M_2**2 / (2 * M**2)
        term3 = M_2**4 / (2 * 2 * M**5)
        term4 = -(M_2**2 * M_3) / (2 * M**4)
        term5 = (x3 - x2 + third) * M_3**2 / (2 * M**3)

        theorem = mpmath.log(prefactor) + (term1 + term2 + term3 + term4 + term5)
        return int(theorem)

    def count_one(self, stoichiometry: AtomStoichiometry):
        return self.count_one_bare(stoichiometry.canonical_tuple)


    def count_one_bare(self, label: tuple[int]):
        try:
            return self._exact_cache[label]
        except:
            pass

        try:
            return self._base_cache[label]
        except:
            pass
    
        degrees = sum([[v]*c for v, c in zip(label[::2], label[1::2])], [])
        if len(degrees) > 20:
            return self._count_one_asymptotically(degrees)

        raise NotImplementedError(
            f"""The pre-computed database does not cover this stoichiometry: {label}.
            
            You may either compute it yourself using build_cache() or contact vonrudorff@uni-kassel.de, 
            so we can distribute the extended cache in a new version of this library, 
            as building the cache is a time-consuming process."""
        )

    @staticmethod
    def sample_connected(spec):
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
    def spec_to_sequence(spec):
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
    def estimate_edit_tree_average_path_length(canonical_label: str, ngraphs: int):

        def _common_input_validation(A, B, partial_match):
            # copied from scipy, since this routine is not exposed but required for custom QAP
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)

            if partial_match is None:
                partial_match = np.array([[], []]).T
            partial_match = np.atleast_2d(partial_match).astype(int)

            msg = None
            if A.shape[0] != A.shape[1]:
                msg = "`A` must be square"
            elif B.shape[0] != B.shape[1]:
                msg = "`B` must be square"
            elif A.ndim != 2 or B.ndim != 2:
                msg = "`A` and `B` must have exactly two dimensions"
            elif A.shape != B.shape:
                msg = "`A` and `B` matrices must be of equal size"
            elif partial_match.shape[0] > A.shape[0]:
                msg = "`partial_match` can have only as many seeds as there are nodes"
            elif partial_match.shape[1] != 2:
                msg = "`partial_match` must have two columns"
            elif partial_match.ndim != 2:
                msg = "`partial_match` must have exactly two dimensions"
            elif (partial_match < 0).any():
                msg = "`partial_match` must contain only positive indices"
            elif (partial_match >= len(A)).any():
                msg = "`partial_match` entries must be less than number of nodes"
            elif not len(set(partial_match[:, 0])) == len(
                partial_match[:, 0]
            ) or not len(set(partial_match[:, 1])) == len(partial_match[:, 1]):
                msg = "`partial_match` column entries must be unique"

            if msg is not None:
                raise ValueError(msg)

            return A, B, partial_match

        def _calc_score(A, B, perm):
            # copied from scipy, since this routine is not exposed but required for custom QAP
            # equivalent to objective function but avoids matmul
            return np.sum(A * B[perm][:, perm])

        def qap_2opt_with_constraints(
            A,
            B,
            groups,
            maximize=False,
            rng=None,
            partial_match=None,
            partial_guess=None,
            **unknown_options,
        ):
            # based on scipy, since this routine is not exposed but required for custom QAP
            _check_unknown_options(unknown_options)
            rng = check_random_state(rng)
            A, B, partial_match = _common_input_validation(A, B, partial_match)

            N = len(A)
            # check trivial cases
            if N == 0 or partial_match.shape[0] == N:
                score = _calc_score(A, B, partial_match[:, 1])
                res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
                return OptimizeResult(res)

            if partial_guess is None:
                partial_guess = np.array([[], []]).T
            partial_guess = np.atleast_2d(partial_guess).astype(int)

            msg = None
            if partial_guess.shape[0] > A.shape[0]:
                msg = (
                    "`partial_guess` can have only as "
                    "many entries as there are nodes"
                )
            elif partial_guess.shape[1] != 2:
                msg = "`partial_guess` must have two columns"
            elif partial_guess.ndim != 2:
                msg = "`partial_guess` must have exactly two dimensions"
            elif (partial_guess < 0).any():
                msg = "`partial_guess` must contain only positive indices"
            elif (partial_guess >= len(A)).any():
                msg = "`partial_guess` entries must be less than number of nodes"
            elif not len(set(partial_guess[:, 0])) == len(
                partial_guess[:, 0]
            ) or not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1]):
                msg = "`partial_guess` column entries must be unique"
            if msg is not None:
                raise ValueError(msg)

            fixed_rows = None
            if partial_match.size or partial_guess.size:
                # use partial_match and partial_guess for initial permutation,
                # but randomly permute the rest.
                guess_rows = np.zeros(N, dtype=bool)
                guess_cols = np.zeros(N, dtype=bool)
                fixed_rows = np.zeros(N, dtype=bool)
                fixed_cols = np.zeros(N, dtype=bool)
                perm = np.zeros(N, dtype=int)

                rg, cg = partial_guess.T
                guess_rows[rg] = True
                guess_cols[cg] = True
                perm[guess_rows] = cg

                # match overrides guess
                rf, cf = partial_match.T
                fixed_rows[rf] = True
                fixed_cols[cf] = True
                perm[fixed_rows] = cf

                random_rows = ~fixed_rows & ~guess_rows
                random_cols = ~fixed_cols & ~guess_cols
                perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
            else:
                perm = rng.permutation(np.arange(N))

            best_score = _calc_score(A, B, perm)

            i_free = np.arange(N)
            if fixed_rows is not None:
                i_free = i_free[~fixed_rows]

            better = operator.gt if maximize else operator.lt
            n_iter = 0
            done = False
            while not done:
                # equivalent to nested for loops i in range(N), j in range(i, N)
                allgroupsdone = True
                for group in groups:
                    if len(group) < 2:
                        continue
                    groupdone = False
                    for i, j in it.combinations(group, 2):
                        n_iter += 1
                        perm[i], perm[j] = perm[j], perm[i]
                        score = _calc_score(A, B, perm)
                        if better(score, best_score):
                            best_score = score
                            break
                        # faster to swap back than to create a new list every time
                        perm[i], perm[j] = perm[j], perm[i]
                    else:  # no swaps made
                        groupdone = True
                    if not groupdone:
                        allgroupsdone = False
                if allgroupsdone:
                    done = True

            res = {"col_ind": perm, "fun": best_score, "nit": n_iter}
            return OptimizeResult(res)

        def _score_permutation(i, j, permutation):
            adj = nx.adjacency_matrix(Gs[j], nodelist=permutation)
            return abs(adjacencies[i] - adj).sum()

        def pair_path_length_QAP_all(
            i,
            j,
        ):
            A = adjacencies[i].todense()
            B = adjacencies[j].todense()

            distances = []
            for _ in range(50):
                permutation = []
                offset = 0
                for element in element_order:
                    nnodes = len(nbe[element])
                    sub_A = A[offset : offset + nnodes, offset : offset + nnodes]
                    sub_B = B[offset : offset + nnodes, offset : offset + nnodes]

                    res = sco.quadratic_assignment(
                        sub_A, sub_B, method="2opt", options={"maximize": True}
                    )

                    permutation += [offset + _ for _ in res.col_ind]
                    offset += nnodes

                distances.append(_score_permutation(i, j, permutation))
            return min(distances)

        def pair_path_length_QAP_blocked(i, j):
            A = adjacencies[i].todense()
            B = adjacencies[j].todense()

            permutation = []
            offset = 0
            for element in element_order:
                nnodes = len(nbe[element])
                sub_A = A[offset : offset + nnodes, offset : offset + nnodes]
                sub_B = B[offset : offset + nnodes, offset : offset + nnodes]

                best_res = None
                for _ in range(20):
                    res = sco.quadratic_assignment(
                        sub_A, sub_B, method="2opt", options={"maximize": True}
                    )
                    if best_res is None or res.fun > best_res.fun:
                        best_res = res

                permutation += [offset + _ for _ in best_res.col_ind]
                offset += nnodes

            return _score_permutation(i, j, permutation)

        def pair_path_length_QAP_constrained(i, j):
            A = adjacencies[i].todense()
            B = adjacencies[j].todense()
            groups = []
            for element in element_order:
                groups.append(nbe[element])

            distances = []
            for _ in range(100):
                res = qap_2opt_with_constraints(A, B, groups, maximize=True)

                distances.append(_score_permutation(i, j, res.col_ind))
            return min(distances)

        def pair_path_length_random_shuffle(i, j):
            distances = []
            for _ in range(100):
                permutation = []
                for element in element_order:
                    copylist = nbe[element].copy()
                    random.shuffle(copylist)
                    permutation += copylist

                distances.append(_score_permutation(i, j, permutation))
            return min(distances)

        def pair_length_random_refinement(i, j):
            distance = None
            groups = [nbe[element].copy() for element in element_order]
            group_weights = np.array([len(_) for _ in groups]) - 1

            for case in range(3):
                for group in groups:
                    random.shuffle(group)
                permutation = sum(groups, [])

                for step in range(30):
                    groupid = rng.choice(
                        len(groups), p=group_weights / sum(group_weights)
                    )
                    offset = sum([len(_) for _ in groups[:groupid]])
                    for nodei, nodej in it.combinations(range(len(groups[groupid])), 2):
                        permutation[offset + nodei], permutation[offset + nodej] = (
                            permutation[offset + nodej],
                            permutation[offset + nodei],
                        )
                        new_distance = _score_permutation(i, j, permutation)
                        if distance is None or new_distance < distance:
                            distance = new_distance
                        else:
                            # swap back
                            permutation[offset + nodei], permutation[offset + nodej] = (
                                permutation[offset + nodej],
                                permutation[offset + nodei],
                            )

            return distance

        def pair_length_all_permutions(i, j):
            distances = []
            for permutation in it.product(*permutations[j]):
                mapping = sum([list(_) for _ in permutation], [])
                distances.append(_score_permutation(i, j, mapping))
            return min(distances)

        def number_of_permutations():
            n = 1
            for element in nbe.values():
                n *= math.factorial(len(element))
            return n

        def nodes_by_element(G):
            elements = {}
            for node, element in nx.get_node_attributes(G, "element").items():
                if element not in elements:
                    elements[element] = []
                elements[element].append(node)
            return elements

        def node_order(G):
            elements = nodes_by_element(G)
            node_order = []
            for element in element_order:
                nodes = elements[element]
                node_order += nodes
            return node_order

        def all_permutations(G):
            elements = nodes_by_element(G)
            permutations = []
            for element in element_order:
                nodes = elements[element]
                permutations.append(list(it.permutations(nodes, len(nodes))))
            return permutations

        Gs = [
            ApproximateCounter.sample_connected(canonical_label) for _ in range(ngraphs)
        ]
        rng = np.random.default_rng()

        # prepare reusable data
        nbe = nodes_by_element(Gs[0])
        element_order = sorted(nbe.keys())

        adjacencies = []
        for G in Gs:
            adjacencies.append(nx.adjacency_matrix(G, nodelist=node_order(G)))
        permutations = None
        if number_of_permutations() < 100:
            permutations = [all_permutations(G) for G in Gs]

        # sum pairwise minimal edit distance
        total_path_length = 0

        strategy_score = []
        for i, j in it.combinations(range(ngraphs), 2):
            strategies = {
                "QB": pair_path_length_QAP_blocked,
                "QA": pair_path_length_QAP_all,
                "QC": pair_path_length_QAP_constrained,
                "RS": pair_path_length_random_shuffle,
                "RR": pair_length_random_refinement,
            }
            if permutations is not None:
                strategies = {"AP": pair_length_all_permutions}

            names = list(strategies.keys())

            distances = [strategies[name](i, j) for name in names]
            total_path_length += min(distances)
            strategy_score.append(names[distances.index(min(distances))])

        strategy_score = collections.Counter(strategy_score)
        return (
            canonical_label,
            ngraphs,
            total_path_length / (ngraphs * (ngraphs - 1) / 2),
            dict(strategy_score),
        )

    @staticmethod
    def _weighted_choose(items: list, weights: list[int]):
        total = sum(weights)
        choice = random.uniform(0, total)
        for item, weight in zip(items, weights):
            choice -= weight
            if choice <= 0:
                return item

    def _sum_formula_database(
        self, search_space: SearchSpace, natoms: int, selection: Q = None
    ):
        sum_formula_size = {}
        stoichiometries = {}
        for stoichiometry in search_space.list_cases(natoms):
            if selection is not None and not selection.selected_stoichiometry(
                stoichiometry
            ):
                continue
            sum_formula = stoichiometry.sum_formula
            magnitude = self.count_one(stoichiometry)
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
        random_order, sizes, stoichiometries = self._sum_formula_database(
            search_space, natoms, selection
        )
        if len(random_order) == 0:
            raise ValueError("Search space and selection yield no feasible molecule.")

        # sample molecules
        molecules = []
        while len(molecules) < nmols:
            sum_formula = ApproximateCounter._weighted_choose(random_order, sizes)
            ss = stoichiometries[sum_formula]
            ws = [self.count_one(_) for _ in ss]
            stoichiometry = ApproximateCounter._weighted_choose(ss, ws)
            spec = stoichiometry.canonical_label

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
        if selection.restricts_bonds:
            raise NotImplementedError(
                "Scoring not available for bond-based selections."
            )

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
