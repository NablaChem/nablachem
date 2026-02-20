"""
Sampling method described in https://arxiv.org/abs/2009.03747
Published as 10.1088/2632-072X/abced5
based on a port of https://github.com/szhorvat/ConnectedGraphSampler
Original code MIT-Licensed Copyright (c) 2020 Szabolcs Horvát
Here: ported to python
"""

import math
import random
from typing import List, Tuple
import networkx as nx


# Type aliases to match C++ code
deg_t = int
edge = Tuple[int, int]
edgelist_t = List[edge]


# Cache for logfact lookup table
_logfact_cache = None


def logfact(n: int) -> float:
    """
    Compute the logarithm of n-factorial.
    """
    global _logfact_cache

    if _logfact_cache is None:
        _logfact_cache = [0.0, 0.0]
        for i in range(2, 257):
            _logfact_cache.append(_logfact_cache[i - 1] + math.log(i))

    if n <= 256:
        return _logfact_cache[n]
    else:
        # Based on: https://www.johndcook.com/blog/2010/08/16/how-to-compute-log-factorial/
        x = n + 1
        return (x - 0.5) * math.log(x) - x + 0.5 * math.log(2 * math.pi) + 1 / (12 * x)


class DegreeSequenceMulti:
    """
    Stores a degree sequence of degrees 0 <= d < n.
    Keeps track of information useful for sampling multigraphs.
    Direct translation from C++ DegreeSequenceMulti class.
    """

    def __init__(self, degrees=None):
        """Initialize degree sequence, O(n)"""
        if degrees is None:
            self.degseq = []
            self.n = 0
            self.dmax = 0
            self.dsum = 0
        else:
            self.degseq = list(degrees)  # Copy the input
            self.n = len(self.degseq)

            if any(d < 0 for d in self.degseq):
                raise ValueError("Degrees must be non-negative.")
            self.dsum = sum(self.degseq)
            self.dmax = max(self.degseq) if self.degseq else 0

    def decrement(self, u: int):
        """Decrement the degree of vertex u, amortized O(1)"""
        self.degseq[u] -= 1
        self.dsum -= 1
        if self.degseq[u] == self.dmax - 1:
            self.dmax = max(self.degseq)

    def connect(self, u: int, v: int):
        """Connect vertices u and v, O(1)"""
        self.decrement(u)
        self.decrement(v)

    def is_multigraphical(self) -> bool:
        """Multigraphicality test, O(1)"""
        return self.dsum % 2 == 0 and self.dsum >= 2 * self.dmax

    def __getitem__(self, v: int) -> deg_t:
        """Access to degrees"""
        return self.degseq[v]

    def __iter__(self):
        """Iterator over degrees"""
        return iter(self.degseq)

    def size(self) -> int:
        """Return number of vertices"""
        return self.n

    def degrees(self) -> List[deg_t]:
        """Return the degree sequence"""
        return self.degseq


class EquivClassElement:
    """
    Used for equivalence class computation.
    Direct translation from C++ EquivClassElement template class.
    """

    def __init__(self):
        self.val = None
        self.equiv = self  # Points to self initially
        self.deg = 0

    def set_value(self, new_val):
        """Set the value of this element"""
        self.val = new_val

    def value(self):
        """Get the value of this element"""
        return self.val

    def set_degree(self, deg_):
        """Set the degree of this element"""
        self.deg = deg_

    def degree(self) -> deg_t:
        """Get the degree of this element"""
        return self.deg

    def get_class_elem(self):
        """
        Get the representative element of this equivalence class.
        Implements path compression for union-find.
        """
        final = self.equiv
        while final != final.equiv:
            final = final.equiv

        # If updating is needed, do a full second pass and update each node of the chain.
        if final != self.equiv:
            e1 = self
            while e1 != e1.equiv:
                e2 = e1.equiv
                e1.equiv = final
                e1 = e2

        return self.equiv

    def update_class(self, elem):
        """
        Update this element to be in the same class as elem.
        """
        new_class = elem.get_class_elem()
        self.get_class_elem().equiv = new_class
        self.equiv = new_class


class EquivClass:
    """
    Track connected components during graph construction.
    Direct translation from C++ EquivClass class.
    """

    def __init__(self, ds):
        """
        Initialize equivalence class tracker with degree sequence.
        Template constructor equivalent taking any container with degrees.
        """
        # Handle both DegreeSequenceMulti objects and plain lists
        self.n = ds.size() if hasattr(ds, "size") else len(ds)
        self.n_supernodes = self.n  # Number of supernodes
        self.n_edges = 0  # Half the number of free stubs
        self.closed = False  # True if degree of supernode dropped to zero before construction complete

        self.elems = [EquivClassElement() for _ in range(self.n)]

        for i, d in enumerate(ds):
            self.elems[i].set_value(i)
            self.elems[i].set_degree(d)
            self.n_edges += d
            if d == 0 and self.n_supernodes != 1:
                self.closed = True

        if self.n_edges % 2 == 1:
            raise ValueError("Connectivity tracker: The degree sum must be even.")
        self.n_edges //= 2

    def connect(self, a: int, b: int):
        """
        Connect vertices a and b, updating component structure.
        """
        self.n_edges -= 1

        class_a = self.elems[a].get_class_elem()
        class_b = self.elems[b].get_class_elem()

        if class_a != class_b:
            self.n_supernodes -= 1

            deg_a = class_a.degree()
            deg_b = class_b.degree()

            self.elems[a].update_class(self.elems[b])

            class_b.set_degree(deg_a + deg_b - 2)
        else:
            class_b.set_degree(class_b.degree() - 2)

        if class_b.degree() == 0 and self.n_edges > 0:
            self.closed = True

    def component_count(self) -> int:
        """Return number of connected components"""
        return self.n_supernodes

    def edge_count(self) -> int:
        """Return number of remaining edges to place"""
        return self.n_edges

    def get_class(self, u: int):
        """Get the equivalence class representative for vertex u"""
        return self.elems[u].get_class_elem()

    def is_potentially_connected(self) -> bool:
        """
        Check if the graph can still become connected.
        """
        return not self.closed and self.n_edges >= self.n_supernodes - 1

    def connectable(self, u: int, v: int) -> bool:
        """
        Returns true if connecting u to v will not break potential connectivity.
        """
        cu = self.get_class(u)
        cv = self.get_class(v)

        cud = cu.degree()
        cvd = cv.degree()

        return (
            self.n_supernodes == 1
            or self.n_edges == 1
            or (cud > 2 and self.n_edges > self.n_supernodes - 1)
            or (cv != cu and (cud > 1 or cvd > 1))
        )


def _discrete_choice(weights, rng=None):
    """
    Choose an index based on weights, equivalent to C++ std::discrete_distribution.
    Direct translation to match C++ behavior exactly.
    """
    if rng is None:
        rng = random.random

    total = sum(weights)
    if total == 0:
        raise ValueError("All weights are zero")

    r = rng() * total
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if r <= cumulative:
            return i
    return len(weights) - 1  # Should not reach here, but just in case


def sample_conn_multi(
    ds: DegreeSequenceMulti, alpha: float, rng=None
) -> Tuple[edgelist_t, float]:
    """
    Sample connected loop-free multigraphs.
    Direct translation from C++ sample_conn_multi function.
    """
    if rng is None:
        rng = random.random

    if not ds.is_multigraphical():
        raise ValueError("The degree sequence is not multigraphical.")

    # The null graph is considered non-connected.
    if ds.n == 0:
        raise ValueError("The degree sequence is not potentially connected.")

    conn_tracker = EquivClass(ds)  # Connectivity tracker
    if not conn_tracker.is_potentially_connected():
        raise ValueError("The degree sequence is not potentially connected.")

    edges = []
    logprob = 0.0

    vertex = 0  # The current vertex that we are connecting up

    # List of vertices that the current vertex can connect to without breaking multigraphicality.
    allowed = []

    # Vertices are chosen with a weight equal to the number of their stubs.
    # This is equivalent to choosing stubs uniformly.
    weights = []

    while True:
        if ds[vertex] == 0:  # No more stubs left on current vertex
            if vertex == ds.n - 1:  # All vertices have been processed
                break

            # Advance to next vertex
            vertex += 1
            continue

        allowed.clear()
        weights.clear()

        # Construct allowed set
        if ds.dsum > 2 * ds.dmax or ds[vertex] == ds.dmax:
            # We can connect to any other vertex

            for v in range(vertex + 1, ds.n):
                if conn_tracker.connectable(vertex, v):
                    allowed.append(v)
                    weights.append(pow(ds[v], alpha))
        else:
            # We can only connect to max degree vertices

            for v in range(vertex + 1, ds.n):
                if ds[v] == ds.dmax:
                    if conn_tracker.connectable(vertex, v):
                        allowed.append(v)
                        weights.append(pow(ds[v], alpha))

        assert len(allowed) > 0, "No allowed connections found"

        tot = sum(weights)
        logprob -= math.log(tot)

        u_index = _discrete_choice(weights, rng)
        u = allowed[u_index]

        logprob += (alpha - 1) * math.log(ds[u])

        ds.connect(u, vertex)
        conn_tracker.connect(u, vertex)
        edges.append((vertex, u))

    # Not all multigraphs correspond to the same number of leaves on the decision tree.
    # Therefore, we must correct the sampling weight.

    multiplicities = {}
    for e in edges:
        multiplicities[e] = multiplicities.get(e, 0) + 1

    for edge_key, multiplicity in multiplicities.items():
        if multiplicity > 1:
            logprob -= logfact(multiplicity)

    return edges, logprob


def generate_connected_multigraph(
    degrees: List[int], alpha: float = 1.0, seed=None
) -> nx.MultiGraph:
    """
    Main interface function to generate a connected loop-free multigraph.

    Args:
        degrees: List of vertex degrees
        alpha: Sampling parameter (default 1.0 for uniform stub sampling)
        seed: Random seed for reproducibility

    Returns:
        NetworkX MultiGraph object representing the generated connected multigraph
    """
    if seed is not None:
        random.seed(seed)

    # Create degree sequence and sample
    ds = DegreeSequenceMulti(degrees)
    edge_list, log_prob = sample_conn_multi(ds, alpha)

    # Create NetworkX MultiGraph
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(degrees)))
    G.add_edges_from(edge_list)

    dv = G.degree
    assert all(dv(n) == deg for n, deg in enumerate(degrees)), "Degree mismatch"

    # Store the log probability as a graph attribute
    G.graph["log_probability"] = log_prob

    return G
