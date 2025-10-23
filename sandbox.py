# %%
from functools import lru_cache
import random
import networkx as nx
import igraph as ig
from collections import Counter
import matplotlib.pyplot as plt

import igraph as ig
from collections import Counter
import matplotlib.pyplot as plt
import pynauty
import igraph as ig
import networkx as nx
from math import prod
from collections import Counter


def aut_size_multigraph_igraph(n_vertices: int, edge_list):
    """
    edge_list: list of (u,v) with repeats for multiplicity (no self-loops).
    Returns |Aut(G)| of the unlabeled multigraph.
    """
    g = ig.Graph()
    # original vertices
    g.add_vertices(n_vertices)
    g.vs["color"] = [0] * n_vertices  # color class 0 = original vertices
    # one vertex per multiedge
    for u, v in edge_list:
        e_idx = g.vcount()
        g.add_vertices(1)
        g.vs[e_idx]["color"] = 1  # color class 1 = edge-vertices
        g.add_edges([(e_idx, u), (e_idx, v)])  # incidence edges
    return g.count_automorphisms()


def aut_size_multigraph_networkx(G: nx.MultiGraph, fix_vertices=False) -> int:
    """
    Wrapper: G is an undirected MultiGraph without self-loops.
    If fix_vertices=True, freezes original vertices (returns only edge-permutation factor).
    """
    # map nodes to 0..n-1
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for (u, v, _) in G.edges(keys=True)]
    if not fix_vertices:
        return aut_size_multigraph_igraph(len(nodes), edges)
    # Freeze vertices: make each original vertex its own color
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["color"] = list(range(len(nodes)))  # unique color per vertex -> fixed
    for u, v in edges:
        e_idx = g.vcount()
        g.add_vertices(1)
        g.vs[e_idx]["color"] = -1  # common color for all edge-vertices
        g.add_edges([(e_idx, u), (e_idx, v)])
    return g.count_automorphisms()


def aut_edge_size_exact(G):
    return aut_size_multigraph_networkx(G, fix_vertices=True)


def enumerate_valid_two_switch_moves(G):
    moves = []
    edges = list(G.edges(keys=True))  # (a,b,k)
    m = len(edges)
    for i in range(m):
        a, b, k1 = edges[i]
        for j in range(i + 1, m):
            c, d, k2 = edges[j]
            if len({a, b, c, d}) < 4:  # 4 distinct endpoints
                continue
            for wiring in (0, 1):  # 0:(a,c)+(b,d), 1:(a,d)+(b,c)
                x1, y1 = (a, c) if wiring == 0 else (a, d)
                x2, y2 = (b, d) if wiring == 0 else (b, c)
                if x1 == y1 or x2 == y2:  # no loops
                    continue
                G.remove_edge(a, b, key=k1)
                G.remove_edge(c, d, key=k2)
                G.add_edge(x1, y1)
                G.add_edge(x2, y2)
                if nx.is_connected(G):
                    moves.append(((a, b, k1), (c, d, k2), wiring))
                G.remove_edge(x1, y1)
                G.remove_edge(x2, y2)
                G.add_edge(a, b, key=k1)
                G.add_edge(c, d, key=k2)
    return moves


def apply_two_switch(G, move):
    (a, b, k1), (c, d, k2), w = move
    Gp = G.copy()
    Gp.remove_edge(a, b, key=k1)
    Gp.remove_edge(c, d, key=k2)
    if w == 0:
        Gp.add_edge(a, c)
        Gp.add_edge(b, d)
    else:
        Gp.add_edge(a, d)
        Gp.add_edge(b, c)
    return Gp


def enumerate_valid_two_switch_results(G):
    results = []
    moves = enumerate_valid_two_switch_moves(G)
    for move in moves:
        Gp = apply_two_switch(G, move)
        results.append(Gp)
    return results


def run_chain(current, steps):
    rng = random.Random(None)
    possible_targets = enumerate_valid_two_switch_moves(current)
    moves_forward = aut_edge_size_exact(current)
    for _ in range(steps):
        proposition = rng.choice(possible_targets)
        proposition = apply_two_switch(current, proposition)
        proposition_targets = enumerate_valid_two_switch_moves(proposition)

        # moves_forward = 0
        # for state in possible_targets:
        #    if nx.isomorphism.is_isomorphic(state, proposition):
        #        moves_forward += 1
        # moves_backward = 0
        # for state in proposition_targets:
        #    if nx.isomorphism.is_isomorphic(state, current):
        #        moves_backward += 1

        # expected = moves_forward / moves_backward
        # alternative = aut_edge_size_exact(current) / aut_edge_size_exact(proposition)
        # if expected != alternative:
        #    print(
        #        moves_forward,
        #        moves_backward,
        #        aut_edge_size_exact(current),
        #        aut_edge_size_exact(proposition),
        #        expected,
        #        alternative,
        #    )
        moves_backward = aut_edge_size_exact(proposition)

        q_fwd = moves_forward / len(possible_targets)
        q_back = moves_backward / len(proposition_targets)
        A = min(1.0, (q_back) / (q_fwd))

        if rng.random() < A:
            current = proposition
            possible_targets = proposition_targets
            moves_forward = moves_backward
    return current


import sys

sys.path.append("/Users/guido/wrk/prototype/randgraph")
import connsampler


import tqdm

degrees = [3] * 4 + [4] * 2
# degrees = [1, 1, 1, 1, 4, 4,4]
nC = 10
nH = nC * 2 + 2
print(nH, nC)
degrees = [4] * nC + [1] * nH


# %%
def do_line(ntotal):
    unique_graphs = {}
    for i in tqdm.trange(ntotal):
        G0 = connsampler.generate_connected_multigraph(degrees)
        G = run_chain(G0, 100)
        for other in unique_graphs.keys():
            if nx.vf2pp_is_isomorphic(G, other, node_label="element"):
                unique_graphs[other] += 1
                break
        else:
            unique_graphs[G] = 1
    total = sum(count for count in unique_graphs.values())
    print(unique_graphs)
    plt.plot(
        sorted([count / total * len(unique_graphs) for count in unique_graphs.values()])
    )


do_line(1000)
# do_line(1000)
# do_line(2000)
plt.axhline(1.0)
# %%
