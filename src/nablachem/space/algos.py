from .utils import Q, SearchSpace, Molecule
import random
from .exact import ExactCounter
from .approximate import ApproximateCounter
import networkx as nx


def chemical_formula_database(
    counter: ExactCounter | ApproximateCounter,
    search_space: SearchSpace,
    natoms: int,
    selection: Q = None,
):
    """Builds a size database of the sizes of all chemical formulas in randomized order.

    Parameters
    ----------
    counter : ExactCounter or ApproximateCounter
        What to use for counting.
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
    for stoichiometry in search_space.list_cases(natoms):
        if selection is not None and not selection.selected_stoichiometry(
            stoichiometry
        ):
            continue
        sum_formula = stoichiometry.sum_formula
        magnitude = counter.count_one(stoichiometry, natoms)
        if sum_formula not in stoichiometries:
            stoichiometries[sum_formula] = [stoichiometry]
            sum_formula_size[sum_formula] = 0
        sum_formula_size[sum_formula] += magnitude

    random_order = list(sum_formula_size.keys())
    random.shuffle(random_order)
    sizes = [sum_formula_size[_] for _ in random_order]

    return random_order, sizes, stoichiometries


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


def random_sample(
    counter: ApproximateCounter,
    search_space: SearchSpace,
    natoms: int,
    nmols: int,
    selection: Q = None,
) -> list[Molecule]:
    """Builds a fixed size random sample from a search space for a given number of atoms.

    Parameters
    ----------
    counter : ApproximateCounter
        What to use for counting.
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
    random_order, sizes, stoichiometries = chemical_formula_database(
        counter, search_space, natoms, selection
    )
    if len(random_order) == 0 or sum(sizes) == 0:
        raise ValueError("Search space and selection yield no feasible molecule.")

    # sample molecules
    molecules = []
    while len(molecules) < nmols:
        sum_formula = _weighted_choose(random_order, sizes)
        ss = stoichiometries[sum_formula]
        ws = [counter.count_one(_, natoms) for _ in ss]
        stoichiometry = _weighted_choose(ss, ws)
        spec = stoichiometry.canonical_label

        # try a few times to find a concrete graph
        # needs to stop eventually, since there might be stoichiometries
        # for which the estimate is non-zero but which are actually
        # infeasible
        tries_selection = 0
        while True and tries_selection < 1e4:
            graph = counter.sample_connected(spec)

            # correct for isomorphisms which skew MC sampling
            degrees, _ = counter.spec_to_sequence(spec)
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
