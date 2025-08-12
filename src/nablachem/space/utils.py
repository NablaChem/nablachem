import collections
import dataclasses
import functools
import itertools as it
import operator
from collections.abc import Iterator

import networkx as nx
import pyparsing
import pysmiles
import tqdm
import gzip
import msgpack


def _read_db(fn: str) -> dict:
    """Reads the database files distributed with the package."""
    with gzip.open(fn) as fh:
        db = msgpack.load(
            fh,
            strict_map_key=False,
            use_list=False,
        )
    return db


@functools.cache
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@functools.cache
def falling_factorial(n, k):
    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result


def integer_partition(total, maxelements):
    """Builds all integer partitions of *total* split into *maxelements* parts."""
    if maxelements == 1:
        yield [total]
        return

    for x in range(total + 1):
        for p in integer_partition(total - x, maxelements - 1):
            yield ([x] + p)


@dataclasses.dataclass(unsafe_hash=True)
class AtomType:
    label: str
    valence: int

    def __post_init__(self):
        if not self.label.isalpha():
            raise ValueError("AtomType label must be alphabetic")


@dataclasses.dataclass
class Element:
    label: str
    valences: list[int]

    @property
    def atom_types(self):
        return [AtomType(self.label, v) for v in self.valences]


@dataclasses.dataclass
class AtomStoichiometry:
    components: dict[AtomType, int] = dataclasses.field(default_factory=dict)

    def extend(self, atom_type: AtomType, count: int):
        self.components[atom_type] = self.components.get(atom_type, 0) + count

    @property
    def _canonical_sorting(self):
        return sorted(self.components.items(), key=lambda x: (x[0].valence, x[1]))

    @property
    def canonical_label(self):
        return "_".join([f"{e[0].valence}.{e[1]}" for e in self._canonical_sorting])

    @property
    def canonical_tuple(self):
        return tuple(
            sum([[e[0].valence, e[1]] for e in self._canonical_sorting if e[1] > 0], [])
        )

    @property
    def canonical_element_sequence(self):
        return tuple(sum([[e[0].label] * e[1] for e in self._canonical_sorting], []))

    @property
    def molecular_label(self):
        sorted_entries = sorted(
            self.components.items(), key=lambda x: (x[0].valence, x[1])
        )
        return "_".join([f"{e[0].label}:{e[0].valence}.{e[1]}" for e in sorted_entries])

    @property
    def sum_formula(self):
        element_count = collections.Counter(
            sum([[atom.label] * count for atom, count in self.components.items()], [])
        )
        element_order = sorted(element_count.keys())
        for prefixed in "HC":
            if prefixed in element_order:
                element_order.remove(prefixed)
                element_order = [prefixed] + element_order

        sumformula = ""
        for element in element_order:
            count = element_count[element]
            sumformula += element
            if count > 1:
                sumformula += str(count)
        return sumformula

    @property
    def num_atoms(self):
        return sum(self.components.values())

    @property
    def largest_valence(self):
        largest = 0
        for atom_type, count in self.components.items():
            if count > 0:
                largest = max(largest, atom_type.valence)
        return largest

    @property
    def degree_sequence(self):
        degrees = []
        for atomtype, natoms in self.components.items():
            degrees += [atomtype.valence] * natoms
        return degrees


class Molecule:
    def __init__(
        self,
        node_labels: list[str],
        edges: list[tuple[int, int, int]] | list[tuple[int, int]],
    ):
        myedges = []
        # convert (idx1, idx2, order) into repeated (idx1, idx2) for networkx
        if len(edges[0]) == 3:
            for edge in edges:
                myedges += [[edge[0], edge[1]]] * edge[2]
        else:
            myedges = edges

        self._G = nx.MultiGraph(myedges)
        nx.set_node_attributes(
            self._G,
            dict(zip(range(len(node_labels)), node_labels)),
            "element",
        )

    def __eq__(self, other):
        return nx.vf2pp_is_isomorphic(self._G, other._G, node_label="element")

    @property
    def connected(self):
        return nx.connected.is_connected(self._G)

    @property
    def SMILES(self):
        mol = nx.Graph()
        for node in self._G.nodes:
            mol.add_node(node, element=self._G.nodes[node]["element"])
        edges = [_[:2] for _ in self._G.edges]
        for bond, order in collections.Counter(edges).items():
            mol.add_edge(bond[0], bond[1], order=order)
        return pysmiles.write_smiles(mol)

    def dumps(self):
        node_labels = nx.get_node_attributes(self._G, "element")
        node_labels = [node_labels[_] for _ in range(len(node_labels))]
        edges = [_[:2] for _ in self._G.edges]
        edges = [
            (bond[0], bond[1], order)
            for bond, order in collections.Counter(edges).items()
        ]
        return node_labels, edges

    def to_stoichiometry(self):
        stoichiometry = AtomStoichiometry()
        for node in self._G.nodes:
            element = self._G.nodes[node]["element"]
            valence = len(list(self._G.edges(node)))
            atomtype = AtomType(element, valence)
            stoichiometry.extend(atomtype, 1)
        return stoichiometry

    def count_bonds(self, element1: str, element2: str, order: int):
        count = 0
        for node in self._G.nodes:
            if self._G.nodes[node]["element"] == element1:
                for neighbor in self._G.neighbors(node):
                    if self._G.nodes[neighbor]["element"] == element2:
                        thisorder = len(list(self._G.edges(node, neighbor)))
                        if thisorder == order:
                            count += 1
        if element1 == element2:
            count //= 2
        return count


def label_to_stoichiometry(label: str):
    parts = label.replace(".", "_").split("_")
    atomtypes = {}
    idx = 0
    for degree, natoms in zip(parts[0::2], parts[1::2]):
        if int(natoms) > 0:
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
