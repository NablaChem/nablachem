from .utils import Q, SearchSpace, Molecule
import random
from .exact import ExactCounter
from .approximate import ApproximateCounter
import tqdm
from ..utils.graph import count_automorphisms


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
    for stoichiometry in search_space.list_cases(natoms, progress=False):
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
    progress: bool = True,
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
    progress : bool, optional
        Show progress bar, by default True

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
    for _ in tqdm.trange(nmols, disable=not progress):
        sum_formula = _weighted_choose(random_order, sizes)
        ss = stoichiometries[sum_formula]
        ws = [counter.count_one(_, natoms) for _ in ss]
        stoichiometry = _weighted_choose(ss, ws)
        spec = stoichiometry.canonical_label

        # random uniform unlabeled graph with given degree sequence
        graph = counter.sample_connected(spec)

        # assign element labels respecting isomorphisms
        underlying_isomorphisms = count_automorphisms(graph, "none")
        while True:
            labels = stoichiometry.atom_types_per_valency.copy()
            for key in labels:
                random.shuffle(labels[key])
            for node in graph.nodes:
                valence = graph.degree[node]
                graph.nodes[node]["element"] = labels[valence].pop()

            this_isomorphisms = count_automorphisms(graph, "element")
            weight = this_isomorphisms / underlying_isomorphisms
            if random.random() < weight:
                break

        mol = Molecule(None, None, graph=graph)
        molecules.append(mol)
    return molecules
