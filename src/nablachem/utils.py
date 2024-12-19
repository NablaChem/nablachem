import dataclasses
import networkx as nx
import collections
import pysmiles


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
        return tuple(sum([[e[0].valence, e[1]] for e in self._canonical_sorting], []))

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
