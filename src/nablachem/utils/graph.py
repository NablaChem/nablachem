import random
import networkx as nx
import igraph as ig
from . import connsampler


def count_automorphisms(G: nx.MultiGraph, node_color_attr: str = None) -> int:
    """Compute the automorphism group size of a multigraph, optionally colored by node attribute."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for (u, v, _) in G.edges(keys=True)]

    g = ig.Graph()
    g.add_vertices(len(nodes))

    if node_color_attr is None:
        node_colors = [0] * len(nodes)
    else:
        mapping = {}
        node_colors = []
        for node in nodes:
            label = G.nodes[node].get(node_color_attr, 0)
            if label not in mapping:
                mapping[label] = len(mapping)
            node_colors.append(mapping[label])

    for u, v in edges:
        e_idx = g.vcount()
        g.add_vertices(1)
        g.add_edges([(e_idx, u), (e_idx, v)])

    all_colors = node_colors + [-1] * len(edges)

    return g.count_automorphisms(color=all_colors)


def _aut_edge_size_exact(G: nx.MultiGraph) -> int:
    """Compute the exact automorphism group size for the edge structure of a multigraph."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for (u, v, _) in G.edges(keys=True)]

    # networkx can enumerate, igraph can count directly
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["color"] = list(range(len(nodes)))
    for u, v in edges:
        e_idx = g.vcount()
        g.add_vertices(1)
        g.vs[e_idx]["color"] = -1
        g.add_edges([(e_idx, u), (e_idx, v)])
    return g.count_automorphisms()


def _enumerate_valid_two_switch_moves(G: nx.MultiGraph) -> list[tuple]:
    moves = []
    edges = list(G.edges(keys=True))
    m = len(edges)
    for i in range(m):
        a, b, k1 = edges[i]
        for j in range(i + 1, m):
            c, d, k2 = edges[j]
            if len({a, b, c, d}) < 4:
                continue
            for wiring in (0, 1):
                x1, y1 = (a, c) if wiring == 0 else (a, d)
                x2, y2 = (b, d) if wiring == 0 else (b, c)
                if x1 == y1 or x2 == y2:
                    continue
                G.remove_edge(a, b, key=k1)
                G.remove_edge(c, d, key=k2)
                G.add_edge(x1, y1)
                G.add_edge(x2, y2)
                # TODO: optimize connectivity check
                # currently simple but overkill: 2-bridge check would be enough
                if nx.is_connected(G):
                    moves.append(((a, b, k1), (c, d, k2), wiring))
                G.remove_edge(x1, y1)
                G.remove_edge(x2, y2)
                G.add_edge(a, b, key=k1)
                G.add_edge(c, d, key=k2)
    return moves


def _apply_two_switch(G, move):
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


def _run_chain(current, steps):
    rng = random.Random(None)
    possible_targets = _enumerate_valid_two_switch_moves(current)
    moves_forward = _aut_edge_size_exact(current)
    for _ in range(steps):
        proposition = rng.choice(possible_targets)
        proposition = _apply_two_switch(current, proposition)
        proposition_targets = _enumerate_valid_two_switch_moves(proposition)
        moves_backward = _aut_edge_size_exact(proposition)

        q_fwd = moves_forward / len(possible_targets)
        q_back = moves_backward / len(proposition_targets)
        A = min(1.0, (q_back) / (q_fwd))

        if rng.random() < A:
            current = proposition
            possible_targets = proposition_targets
            moves_forward = moves_backward
    return current


def random_connected_graph(degrees: list[int]):
    """Generates a random connected graph with the given degree sequence."""
    # get any connected graph with the given degree sequence
    G0 = connsampler.generate_connected_multigraph(degrees)
    # perform MCMC to make it uniformly random
    return _run_chain(G0, 100)
