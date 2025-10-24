import nablachem.space.utils as ncu
import pytest
import networkx as nx
from nablachem.utils.graph import count_automorphisms


def test_atomtype_label_simple():
    with pytest.raises(ValueError):
        ncu.AtomType(label="something with space", valence=1)

    with pytest.raises(ValueError):
        ncu.AtomType(label="with<special>characters", valence=1)


def test_automorphisms_H2():
    case = nx.MultiGraph()
    case.add_node(0, element="H")
    case.add_node(1, element="H")
    case.add_edge(0, 1)

    assert count_automorphisms(case, "element") == 2


def test_automorphisms_C2H4():
    case_linear = nx.MultiGraph()
    case_linear.add_node(0, element="C")
    case_linear.add_node(1, element="C")
    case_linear.add_node(2, element="H")
    case_linear.add_node(3, element="H")
    case_linear.add_node(4, element="H")
    case_linear.add_node(5, element="H")
    case_linear.add_edge(0, 1)
    case_linear.add_edge(0, 2)
    case_linear.add_edge(0, 3)
    case_linear.add_edge(1, 4)
    case_linear.add_edge(1, 5)

    assert count_automorphisms(case_linear, "element") == 8


def test_automorphism_C2H2F2():
    case = nx.MultiGraph()
    case.add_node(0, element="C")
    case.add_node(1, element="C")
    case.add_node(2, element="H")
    case.add_node(3, element="H")
    case.add_node(4, element="F")
    case.add_node(5, element="F")
    case.add_edge(0, 1)
    case.add_edge(0, 2)
    case.add_edge(0, 4)
    case.add_edge(1, 3)
    case.add_edge(1, 5)

    assert count_automorphisms(case, "element") == 2

    case = nx.MultiGraph()
    case.add_node(0, element="C")
    case.add_node(1, element="C")
    case.add_node(2, element="H")
    case.add_node(3, element="H")
    case.add_node(4, element="F")
    case.add_node(5, element="F")
    case.add_edge(0, 1)
    case.add_edge(0, 2)
    case.add_edge(0, 3)
    case.add_edge(1, 4)
    case.add_edge(1, 5)

    assert count_automorphisms(case, "element") == 4


def test_automorphisms_no_label():
    case = nx.MultiGraph()
    case.add_node(0, element="C")
    case.add_node(1, element="C")
    case.add_node(2, element="H")
    case.add_node(3, element="H")
    case.add_node(4, element="F")
    case.add_node(5, element="F")
    case.add_edge(0, 1)
    case.add_edge(0, 2)
    case.add_edge(0, 3)
    case.add_edge(1, 4)
    case.add_edge(1, 5)

    assert count_automorphisms(case, "nonexisting") == 8
