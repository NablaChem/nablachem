import collections
import itertools as it
import math
import operator
import random

import networkx as nx
import numpy as np
import scipy.optimize as sco
from scipy._lib._util import check_random_state
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options

import nablachem.space as ncs
import networkx as nx
from collections import defaultdict, Counter
from .approximate import ApproximateCounter


def estimate_edit_tree_average_path_length(canonical_label: str, ngraphs: int = 5):
    one = _permutation_free(canonical_label, ngraphs)
    two = _with_permutations(canonical_label, ngraphs)[2]
    return min(one, two)


def _permutation_free(canonical_label: str, ngraphs: int = 5):
    def canonicalize_by_spectral_ordering(G):
        order = list(nx.spectral_ordering(G))  # returns nodes in spectral order
        mapping = {node: i for i, node in enumerate(order)}
        return nx.relabel_nodes(G, mapping, copy=True)

    def extract_core_graph(G):
        G_core = G.copy()

        deg1_nodes = [n for n in G.nodes if G.degree(n) == 1]
        element_count = Counter(G.nodes[n]["element"] for n in deg1_nodes)

        if not element_count:
            return G_core

        most_common_element, _ = element_count.most_common(1)[0]

        terminal_counts = defaultdict(int)

        for n in deg1_nodes:
            if G.nodes[n]["element"] != most_common_element:
                continue
            neighbors = list(G.neighbors(n))
            if len(neighbors) != 1:
                continue  # safety
            parent = neighbors[0]
            terminal_counts[parent] += 1
            G_core.remove_node(n)

        for n in G_core.nodes:
            original = G_core.nodes[n]["element"]
            count = terminal_counts.get(n, 0)
            if count > 0:
                G_core.nodes[n]["element"] = f"{original}+{count}"

        return G_core

    Gs = []
    for _ in range(ngraphs):
        G = ncs.ApproximateCounter.sample_connected(canonical_label)
        G = extract_core_graph(G)
        G = canonicalize_by_spectral_ordering(G)
        Gs.append(G)

    Gsmod = []
    # make node insertion follow the canonical order, does not change the graph but helps optimize_edit_paths
    for G in Gs:
        H = nx.MultiGraph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        Gsmod.append(H)

    # get pairwise distances
    costs = []
    for i in range(ngraphs):
        for j in range(i + 1, ngraphs):
            # shortcut if they are isomorphic
            if nx.is_isomorphic(
                Gsmod[i],
                Gsmod[j],
                node_match=lambda n1, n2: n1["element"] == n2["element"],
            ):
                costs.append(0)
                continue
            matcher = nx.optimize_edit_paths(
                Gsmod[i],
                Gsmod[j],
                node_match=lambda n1, n2: n1["element"] == n2["element"],
                node_del_cost=lambda _: float("inf"),
                node_ins_cost=lambda _: float("inf"),
                edge_del_cost=lambda e: 1,
                edge_ins_cost=lambda e: 1,
                timeout=25,
                strictly_decreasing=True,
            )
            for _ in range(20):
                try:
                    node_edits, _, cost = next(matcher)
                    node_edits = [_ for _ in node_edits if _[0] != _[1]]
                    # print(cost, node_edits)
                except StopIteration:
                    break
            costs.append(cost)
    return canonical_label, sum(costs) / len(costs)


def _with_permutations(canonical_label: str, ngraphs: int):
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
        elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not len(
            set(partial_match[:, 1])
        ) == len(partial_match[:, 1]):
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
            msg = "`partial_guess` can have only as " "many entries as there are nodes"
        elif partial_guess.shape[1] != 2:
            msg = "`partial_guess` must have two columns"
        elif partial_guess.ndim != 2:
            msg = "`partial_guess` must have exactly two dimensions"
        elif (partial_guess < 0).any():
            msg = "`partial_guess` must contain only positive indices"
        elif (partial_guess >= len(A)).any():
            msg = "`partial_guess` entries must be less than number of nodes"
        elif not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or not len(
            set(partial_guess[:, 1])
        ) == len(partial_guess[:, 1]):
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

            for step in range(300):
                groupid = rng.choice(len(groups), p=group_weights / sum(group_weights))
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

    Gs = [ApproximateCounter.sample_connected(canonical_label) for _ in range(ngraphs)]

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
