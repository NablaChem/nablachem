# %%
# Illustrates how to run a new average path length heuristic to improve estimates
import nablachem.space as ncs
import networkx as nx
from collections import defaultdict, Counter
import hmq


def extract_core_graph(G):
    G_core = G.copy()

    deg1_nodes = [n for n in G.nodes if G.degree(n) == 1]
    element_count = Counter(G.nodes[n]["element"] for n in deg1_nodes)

    if not element_count:
        return G_core  # nothing to do

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


@hmq.task
def new_strategy(canonical_label: str, ngraphs: int = 5):
    # sample random connected

    def canonicalize_by_spectral_ordering(G):
        order = list(nx.spectral_ordering(G))  # returns nodes in spectral order
        mapping = {node: i for i, node in enumerate(order)}
        return nx.relabel_nodes(G, mapping, copy=True)

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
    # print(costs)
    return canonical_label, sum(costs) / len(costs)


# 1.13_3.3_4.8: with core 12.5 20993942
# 1.20_4.1_5.8: with core 10.7 37

# %%
G = ncs.ApproximateCounter.sample_connected("1.13_3.3_4.8")

G2 = ncs.ApproximateCounter.sample_connected("1.13_3.3_4.8")
# %%
# G2 = ncs.ApproximateCounter.sample_connected("1.2_2.3")
import networkx as nx
from networkx.algorithms.isomorphism import MultiGraphMatcher
from collections import Counter


def relabel_multiedges(G, mapping):
    """Return a multiset (Counter) of edges after applying node relabeling."""
    edges = []
    for u, v, _ in G.edges(keys=True):
        a, b = mapping[u], mapping[v]
        edges.append(tuple(sorted((a, b))))
    return Counter(edges)


def multigraph_edit_distance_with_node_attr(G_small, G_large):
    if G_small.number_of_nodes() > G_large.number_of_nodes():
        G_small, G_large = G_large, G_small

    GM = MultiGraphMatcher(
        G_large, G_small, node_match=lambda a, b: a["element"] == b["element"]
    )
    min_ged = float("inf")

    E_large = Counter(tuple(sorted((u, v))) for u, v, _ in G_large.edges(keys=True))

    for mapping in GM.isomorphisms_iter():
        mapped_edges = relabel_multiedges(G_small, mapping)
        sym_diff = (E_large - mapped_edges) + (mapped_edges - E_large)
        ged = sum(sym_diff.values()) // 2
        min_ged = min(min_ged, ged)

    return min_ged


def ref(G1, G2):
    matcher = nx.optimize_edit_paths(
        G1,
        G2,
        node_match=lambda n1, n2: n1["element"] == n2["element"],
        node_del_cost=lambda _: float("inf"),
        node_ins_cost=lambda _: float("inf"),
        edge_del_cost=lambda e: 1,
        edge_ins_cost=lambda e: 1,
        timeout=3000,
        strictly_decreasing=True,
    )
    for _ in range(20):
        try:
            node_edits, _, cost = next(matcher)
            node_edits = [_ for _ in node_edits if _[0] != _[1]]
            print(cost, node_edits)
        except StopIteration:
            break
    return cost


multigraph_edit_distance_with_node_attr(extract_core_graph(G), extract_core_graph(G2))


# ref(extract_core_graph(G), extract_core_graph(G2))
# %%
def showme(G):
    labels = {n: d["element"] for n, d in G.nodes(data=True)}
    nx.draw(G, with_labels=True, node_color="red", labels=labels)


(1, 3, 1, 10, 5, 5),
G = ncs.ApproximateCounter.sample_connected("1.3_1.10_5.5")
showme(G)


# %%
import gzip
import msgpack


def read_file(fn):
    with gzip.open(fn) as fh:
        db = msgpack.load(
            fh,
            strict_map_key=False,
            use_list=False,
        )
    for k in db:
        valences, freqs = k[::2], k[1::2]
        parts = []
        for valence, freq in zip(valences, freqs):
            parts.append(f"{valence}.{freq}")

        label = "_".join(parts)
        new_strategy(label)


read_file("/Users/guido/wrk/nablachem/maintenance/space-approx.msgpack.gz")
new_strategy.submit(tag="approx")
# %%
new_strategy.submit(tag="compare")

# %%
with open("/Users/guido/wrk/nablachem/maintenance/space-away.txt") as fh:
    for line in fh:
        label = line.strip()
        new_strategy(label)
new_strategy.submit(tag="compare5_refine")
# %%
import hmq

tag = hmq.Tag.from_file(
    "/Users/guido/wrk/nablachem/maintenance/approx-better-strategy.tag"
)
# %%
cases = []
for r in tag.results:
    if r:
        cases.append(f"{r[0]} {r[1]}")
with open("/Users/guido/wrk/nablachem/database/space-approx-refined.txt", "w") as fh:
    fh.write("\n".join(cases))

# %%
