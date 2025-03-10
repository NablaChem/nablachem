import subprocess
from collections.abc import Iterator
import scipy.special as scs
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
import tqdm
import gzip
from mpmath import mp
import mpmath
from .utils import *


def label_to_stoichiometry(label: str):
    parts = label.replace(".", "_").split("_")
    atomtypes = {}
    idx = 0
    for degree, natoms in zip(parts[0::2], parts[1::2]):
        atomtypes[AtomType("X" + chr(ord("a") + idx), int(degree))] = int(natoms)
        idx += 1
    return AtomStoichiometry(atomtypes)


def _is_pure(label: tuple[int]) -> bool:
    """Tests whether a colored degree sequence is pure.

    A pure degree sequence has no two colors ("elements") with same valency.

    Parameters
    ----------
    label : tuple[int]
        (degree, natoms, degree, natoms)

    Returns
    -------
    bool
        Whether the sequence is pure.
    """
    valences = label[::2]
    if len(valences) == len(set(valences)):
        return True
    return False


def _to_pure(label: tuple[int]) -> tuple[int]:
    """Finds the corresponding pure colored degree sequence from a (potentially) non-pure one.

    Parameters
    ----------
    label : tuple[int]
        (degree, natoms, degree, natoms)

    Returns
    -------
    tuple[int]
        (degree, natoms, degree, natoms)
    """

    purespec = []

    last_d = label[0]
    counts = []
    for i in range(len(label) // 2):
        degree, count = label[i * 2 : i * 2 + 2]
        if degree != last_d:
            counts = sum(counts)
            purespec += [last_d, counts]
            last_d = degree
            counts = []
        counts.append(count)
    counts = sum(counts)
    purespec += [last_d, counts]
    return tuple(purespec)


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
        progress: bool = True,
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
        progress : bool, optional
            Whether to show a progress bar, by default True

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

        outer_partitions = list(integer_partition(natoms, len(valences)))

        if progress:
            washlist = tqdm.tqdm(outer_partitions, desc="Partition over valences")
        else:
            washlist = outer_partitions
        for outer_partition in washlist:
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

            yield None  # denotes a new degree sequence
            for case in it.product(*groups):
                yield sum(case, [])

    def list_cases(
        self, natoms: int, progress: bool = True
    ) -> Iterator[AtomStoichiometry]:
        for case in self.list_cases_bare(natoms, progress=progress):
            if case is None or case == []:
                continue
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
    def covered_search_space(kind: str):
        """Returns the pre-defined chemical spaces from the original publication

        Parameters
        ----------
        kind : str
            Label, either A or B.

        Returns
        -------
        SearchSpace
            The chosen space.
        """
        if kind not in ["A", "B"]:
            raise ValueError("No such label.")

        s = SearchSpace()
        s.add_element(Element("C", [4]))
        s.add_element(Element("O", [2]))
        s.add_element(Element("F", [1]))
        s.add_element(Element("H", [1]))
        s.add_element(Element("Cl", [1]))
        s.add_element(Element("Br", [1]))
        s.add_element(Element("I", [1]))
        s.add_element(Element("P", [3, 5]))
        if kind == "A":
            s.add_element(Element("N", [3, 5]))
            s.add_element(Element("S", [2, 4, 6]))
            s.add_element(Element("Si", [4]))
        else:
            s.add_element(Element("N", [3]))

        return s


class Q:
    def __init__(self, query_string: str):
        self._parsed = Q._parse_query(query_string)

    @staticmethod
    def _parse_query(query_string: str):
        if query_string.strip() == "":
            return []

        alpha_identifier = pyparsing.Word(pyparsing.alphas)
        hash_identifier = pyparsing.Literal("#")
        identifier = alpha_identifier | hash_identifier

        number = pyparsing.Word(pyparsing.nums)
        operand = identifier | number

        comparison_op = pyparsing.oneOf("< > = <= >= !=")
        and_op = pyparsing.oneOf("and &", caseless=True)
        or_op = pyparsing.oneOf("or |", caseless=True)
        not_op = pyparsing.oneOf("not !", caseless=True)
        add_op = pyparsing.oneOf("+ -")

        arithmetic_expr = pyparsing.infixNotation(
            operand,
            [
                (add_op, 2, pyparsing.opAssoc.LEFT),
            ],
        )

        comparison_expr = pyparsing.Group(
            arithmetic_expr + comparison_op + arithmetic_expr
        )
        not_expr = pyparsing.Group(not_op + comparison_expr)
        parser = pyparsing.infixNotation(
            comparison_expr | not_expr,
            [
                (not_op, 1, pyparsing.opAssoc.RIGHT),
                (and_op, 2, pyparsing.opAssoc.LEFT),
                (or_op, 2, pyparsing.opAssoc.LEFT),
            ],
        )
        try:
            return parser.parseString(query_string, parseAll=True).as_list()
        except:
            raise ValueError(
                "Cannot parse query string. Mismatched parentheses or invalid syntax?"
            )

    def selected_stoichiometry(
        self, stoichiometry: AtomStoichiometry | list[str]
    ) -> bool:
        if isinstance(stoichiometry, AtomStoichiometry):
            element_counts = collections.Counter(
                stoichiometry.canonical_element_sequence
            )
        else:
            element_counts = collections.Counter(stoichiometry)
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
            "!": operator.not_,
            "+": operator.add,
            "-": operator.sub,
        }

        def evaluate(parsed):
            if parsed == []:
                return True

            if parsed in (True, False):
                return parsed

            if isinstance(parsed, int):
                return parsed

            if isinstance(parsed, str):
                try:
                    return int(parsed)
                except:
                    return element_counts[parsed]

            if len(parsed) == 1:
                return evaluate(parsed[0])

            if len(parsed) == 2:
                op, rhs = parsed
                return operators[op](evaluate(rhs))

            if len(parsed) > 3:
                ops = parsed[1::2]
                # and precedence
                anyof = ("|", "or")
                if "&" in ops or "and" in ops:
                    anyof = ("&", "and")
                try:
                    opidx = ops.index(anyof[0])
                except:
                    try:
                        opidx = ops.index(anyof[1])
                    except:
                        opidx = 0
                opidx = opidx * 2 + 1
                center = evaluate(parsed[opidx - 1 : opidx + 2])
                left = parsed[: opidx - 1]
                right = parsed[opidx + 2 :]
                return evaluate(left + [center] + right)

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
                res = operators[op](lhs, rhs)
                return res
            else:
                lhs = evaluate(lhs)
                rhs = evaluate(rhs)
                res = operators[op](lhs, rhs)
                return res

        return evaluate(self._parsed)

    def selected_molecule(self, mol: Molecule):
        # should be overridden to implement rejection sampling
        return True


class ExactCounter:
    """Python API for exact counts of molecular graphs via surge.

    Note that this implementation avoids the built-in pruning of
    "infeasible" molecular graphs in surge by defining non-standard element labels.

    Uses surge (https://doi.org/10.1186/s13321-022-00604-9) which in turn leverages nauty.
    """

    def __init__(self, binary: str, timeout: int = None):
        """Sets up the environment.

        Parameters
        ----------
        binary : str
            Path to the surge binary.
        timeout : int, optional
            Limits the total runtime for counting any one chemical formula, by default None
        """
        self._binary = binary
        self._cache = {}
        self._timeout = timeout

    def _build_cli_arguments(
        self, stoichiometry: AtomStoichiometry, count_only: bool
    ) -> tuple[str, dict[str, str]]:
        max_valence = stoichiometry.largest_valence
        args = [f"-c{max_valence}", f"-d{max_valence}"]
        sf = ""

        letter = "a"
        elements = {}
        used_hydrogen = False
        for atom_type, natoms in stoichiometry.components.items():
            valence = atom_type.valence
            if valence == 1 and not used_hydrogen and count_only:
                elements["H"] = atom_type.label
                sf += f"H{natoms}"
                used_hydrogen = True
            else:
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
        stdout = subprocess.check_output(
            cmd, shell=True, stderr=stderr, timeout=self._timeout
        )
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

    def list_one(self, stoichiometry: AtomStoichiometry) -> Iterator[Molecule]:
        args, lookup = self._build_cli_arguments(stoichiometry, count_only=False)

        stdout = self._run(args, keep_stderr=False)

        # node labels
        node_labels = []
        for component in stoichiometry.components.items():
            node_labels += [component[0].label] * component[1]

        bondtypes = {"-": 1, "=": 2, "#": 3}
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

            yield Molecule(elements, edges)

    def list(
        self, search_space: SearchSpace, natoms: int, selection: Q = None
    ) -> Iterator[Molecule]:
        for stoichiometry in search_space.list_cases(natoms):
            if selection.selected_stoichiometry(stoichiometry):
                yield from self.list_one(stoichiometry)


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
                total += self.count_one(stoichiometry, natoms)
        else:
            cached_degree_sequence = False
            for case in search_space.list_cases_bare(natoms, progress=self._progress):
                if case is None:
                    cached_degree_sequence = False
                    continue

                components = [[valence, count] for _, valence, count in case]
                total += self.count_one_bare(
                    tuple(sum(components, [])), natoms, cached_degree_sequence
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

        # calibration via error term, see SI
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

    def count_one(self, stoichiometry: AtomStoichiometry, natoms: int) -> int:
        """Counts the total number of molecules in a given stoichiometry.

        The redundant specification of the number of atoms is a performance tweak.

        Parameters
        ----------
        stoichiometry : AtomStoichiometry
            The stoichiometry to count.
        natoms : int
            Number of atoms in that stoichiometry.

        Returns
        -------
        int
            Total count of molecules.
        """
        self._fill_cache(natoms)
        return self.count_one_bare(stoichiometry.canonical_tuple, natoms)

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

        M = 0
        purespec = []
        logscore = 0

        last_d = label[0]
        counts = []
        for i in range(len(label) // 2):
            degree, count = label[i * 2 : i * 2 + 2]
            if degree != last_d:
                logscore += self._cached_permutation_factor_log(tuple(counts))
                counts = sum(counts)
                purespec += [last_d, counts]
                M += last_d * counts
                last_d = degree
                counts = []
            counts.append(count)
        logscore += self._cached_permutation_factor_log(tuple(counts))
        counts = sum(counts)
        purespec += [last_d, counts]
        purespec = tuple(purespec)
        M += last_d * counts

        if pure_size is None:
            if log_size is None:
                pure_size = _from_cache()
            else:
                pure_size = int(mpmath.exp(log_size))

        prefactor = logscore / M + 1
        lgdu = self._size_to_average_path_length(pure_size)
        return self._average_path_length_to_size(prefactor * lgdu)

    def count_one_bare(
        self, label: tuple[int], natoms: int, cached_degree_sequence: bool = False
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

        Returns
        -------
        int
            Total count.
        """
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
    def estimate_edit_tree_average_path_length(canonical_label: str, ngraphs: int):
        """Estimates the average path length via graph edit distance heuristics.

        This is done by choosing `ngraphs` random molecules and calculating their pairwise graph distance
        by finding the minimal edit distance between them. The final answer is then averaged over all unique
        pairs in this list. Therefore runtime scales quadratically with `ngraphs`.

        The shorted edit distance is defined as the Wasserstein metric between adjacency matrices of
        two molecules.

        The function also returns a statistic of the success of the different strategies to propose the
        minimal edit distance. For a description of those strategies, see the docstrings of the local functions.

        canonical_label,
            ngraphs,
            total_path_length / (ngraphs * (ngraphs - 1) / 2),
            dict(strategy_score),

        Parameters
        ----------
        canonical_label : str
            The case for which to run the computation. Format is "degree.natoms_degree.natoms_degree.natoms".
        ngraphs : int
            Number of graphs to use for averaging. Suggested to be 30-50.

        Returns
        -------
        tuple[str, int, float, dict[str, int]]
            The canonical label, the number of graphs, the average path length, the success counts of the individual strategies.
        """

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

        def _score_permutation(i: int, j: int, permutation: list[int]) -> float:
            """Calculates the minimum edit distance as the Wasserstein metric between
            adjacency matrices of two molecules after a permutation has been applied to
            molecule j.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.
            permutation : list[int]
                Reordering of atoms.

            Returns
            -------
            float
                Distance.
            """
            adj = nx.adjacency_matrix(Gs[j], nodelist=permutation)
            return abs(adjacencies[i] - adj).sum()

        def pair_path_length_QAP_all(
            i,
            j,
        ):
            """Tries to improve the edit distance between two molecules by considering it a
            quadratic assignment problem within each block of element identities sequentially
            in multiple iterations.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.

            Returns
            -------
            float
                The minimal distance.
            """
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
            """Tries to improve the edit distance between two molecules by considering it a
            quadratic assignment problem within each block of element identities individually.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.

            Returns
            -------
            float
                The minimal distance.
            """
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
            """Tries to improve the edit distance between two molecules by considering it a
            quadratic assignment problem in global blocks of element identities.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.

            Returns
            -------
            float
                The minimal distance.
            """
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
            """Tries to improve the edit distance between two molecules by random shuffles of atoms of same elements.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.

            Returns
            -------
            float
                The minimal distance.
            """
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
            """Tries to improve the edit distance between two molecules by random exchanges of atoms of same elements.

            Parameters
            ----------
            i : int
                Molecule index.
            j : int
                Molecule index.

            Returns
            -------
            float
                The minimal distance.
            """
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
            """Finds the shortest distance for an exhaustive permutation of atom assignments
            which keep the atom identity unchanged.

            Parameters
            ----------
            i : int
                Molecule index
            j : int
                Molecule index

            Returns
            -------
            float
                Minimum distance (Wasserstein metric)
            """
            distances = []
            for permutation in it.product(*permutations[j]):
                mapping = sum([list(_) for _ in permutation], [])
                distances.append(_score_permutation(i, j, mapping))
            return min(distances)

        def number_of_permutations():
            """Calculates the total number of permutations of atoms without changing atom identity."""
            n = 1
            for element in nbe.values():
                n *= math.factorial(len(element))
            return n

        def nodes_by_element(G):
            """Groups atoms by their element label."""
            elements = {}
            for node, element in nx.get_node_attributes(G, "element").items():
                if element not in elements:
                    elements[element] = []
                elements[element].append(node)
            return elements

        def node_order(G):
            """Create a static order where the graph node indices are sorted by element.
            The order within each element is undefined."""
            elements = nodes_by_element(G)
            node_order = []
            for element in element_order:
                nodes = elements[element]
                node_order += nodes
            return node_order

        def all_permutations(G):
            """Builds an explicit list of all permutations of all atoms within each element."""
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
