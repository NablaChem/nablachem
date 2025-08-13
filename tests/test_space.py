import pyparsing
import pytest
import nablachem.space as ncs
import numpy as np


@pytest.fixture(scope="session")
def approximate_counter():
    """Shared ApproximateCounter instance for all tests."""
    return ncs.ApproximateCounter()


def test_case_noatoms():
    s = ncs.SearchSpace()
    s.add_element(ncs.Element("C", [4]))
    s.add_element(ncs.Element("F", [1]))
    s.add_element(ncs.Element("H", [1]))
    assert len(list(s.list_cases(0))) == 0


def test_case_list():
    s = ncs.SearchSpace()
    s.add_element(ncs.Element("C", [4]))
    s.add_element(ncs.Element("F", [1]))
    s.add_element(ncs.Element("H", [1]))

    expected = [
        ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 3})
    ]
    assert list(s.list_cases(3)) == expected

    expected = [
        ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4}),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 2,
                ncs.AtomType(label="H", valence=1): 2,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 2,
                ncs.AtomType(label="F", valence=1): 2,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 2,
                ncs.AtomType(label="H", valence=1): 1,
                ncs.AtomType(label="F", valence=1): 1,
            }
        ),
    ]
    actual = list(s.list_cases(4))
    for a in actual:
        assert a in expected
    assert len(actual) == len(expected)

    expected = [
        ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 5}),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="F", valence=1): 4,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="H", valence=1): 4,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="F", valence=1): 3,
                ncs.AtomType(label="H", valence=1): 1,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="F", valence=1): 2,
                ncs.AtomType(label="H", valence=1): 2,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="F", valence=1): 1,
                ncs.AtomType(label="H", valence=1): 3,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="H", valence=1): 2,
                ncs.AtomType(label="C", valence=4): 3,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="F", valence=1): 2,
                ncs.AtomType(label="C", valence=4): 3,
            }
        ),
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="H", valence=1): 1,
                ncs.AtomType(label="F", valence=1): 1,
                ncs.AtomType(label="C", valence=4): 3,
            }
        ),
    ]

    actual = list(s.list_cases(5))
    for a in actual:
        assert a in expected
    assert len(actual) == len(expected)


def _compare_caselists(actual, expected):
    # sorting within one case does not matter, sorting of cases does,
    # but only between None separators which indicate a new degree sequence starting
    if len(actual) != len(expected):
        return False

    def _to_blocks(caselist):
        blocks = []
        block = []
        for entry in caselist:
            if entry is None:
                blocks.append(set(block))
                block = []
            else:
                block.append(tuple(sorted(entry)))
        blocks.append(set(block))
        return blocks

    blocks_actual = _to_blocks(actual)
    blocks_expected = _to_blocks(expected)
    if len(blocks_actual) != len(blocks_expected):
        return False

    for block in blocks_expected:
        index = blocks_actual.index(block)
        del blocks_actual[index]
    return True


def test_case_list_bare():
    s = ncs.SearchSpace()
    s.add_element(ncs.Element("C", [4]))
    s.add_element(ncs.Element("F", [1]))
    s.add_element(ncs.Element("H", [1]))

    expected = [None, [("C", 4, 3)]]
    assert list(s.list_cases_bare(3)) == expected

    expected = [
        None,
        [("C", 4, 4)],
        None,
        [("C", 4, 2), ("H", 1, 2)],
        [("C", 4, 2), ("F", 1, 2)],
        [("C", 4, 2), ("H", 1, 1), ("F", 1, 1)],
    ]
    actual = list(s.list_cases_bare(4))

    assert _compare_caselists(actual, expected)

    expected = [
        None,
        [("C", 4, 5)],
        None,
        [("H", 1, 2), ("C", 4, 3)],
        [("F", 1, 2), ("C", 4, 3)],
        [("H", 1, 1), ("F", 1, 1), ("C", 4, 3)],
        None,
        [("C", 4, 1), ("F", 1, 4)],
        [("C", 4, 1), ("H", 1, 4)],
        [("C", 4, 1), ("F", 1, 3), ("H", 1, 1)],
        [("C", 4, 1), ("F", 1, 2), ("H", 1, 2)],
        [("C", 4, 1), ("F", 1, 1), ("H", 1, 3)],
    ]
    actual = list(s.list_cases_bare(5))

    assert _compare_caselists(actual, expected)


def test_canonical_label(approximate_counter):
    for input, expected in (
        ((1, 30, 4, 9), (1, 30, 4, 9)),
        ((4, 9, 1, 30), (1, 30, 4, 9)),
        ((4, 5, 4, 4, 1, 5, 1, 25), (1, 5, 1, 25, 4, 4, 4, 5)),
    ):
        assert approximate_counter._canonical_label(input) == expected


def test_count_unvalidated(approximate_counter):
    assert (
        approximate_counter.count_one(
            ncs.label_to_stoichiometry("1.30_4.9"), 9 + 30, validated=True
        )
        > 0
    )
    assert (
        approximate_counter.count_one(ncs.label_to_stoichiometry("1.30_4.9"), 9 + 30)
        == 0
    )
    assert (
        approximate_counter.count_one(ncs.label_to_stoichiometry("1.1_4.9"), 9 + 30)
        == 0
    )


def test_zero_frequency(approximate_counter):
    assert approximate_counter.count_one(ncs.label_to_stoichiometry("1.0_4.9"), 9) > 0


def test_case_list_bare_sequence():
    s = ncs.SearchSpace()
    s.add_element(ncs.Element("C", [4]))
    s.add_element(ncs.Element("F", [1]))
    s.add_element(ncs.Element("H", [1]))

    expected = [None, [(4, 3)]]
    assert list(s.list_cases_bare(3, degree_sequences_only=True)) == expected

    expected = [
        None,
        [(4, 4)],
        None,
        [(4, 2), (1, 2)],
        [(4, 2), (1, 1), (1, 1)],
    ]
    actual = list(s.list_cases_bare(4, degree_sequences_only=True))

    assert _compare_caselists(actual, expected)

    expected = [
        None,
        [(4, 5)],
        None,
        [(4, 1), (1, 4)],
        [(4, 1), (1, 3), (1, 1)],
        [(4, 1), (1, 2), (1, 2)],
        None,
        [(1, 2), (4, 3)],
        [(1, 1), (1, 1), (4, 3)],
    ]
    actual = list(s.list_cases_bare(5, degree_sequences_only=True))

    assert _compare_caselists(actual, expected)


def test_case_list_bare_sequence_pure():
    s = ncs.SearchSpace()
    s.add_element(ncs.Element("C", [4]))
    s.add_element(ncs.Element("F", [1]))
    s.add_element(ncs.Element("H", [1]))

    expected = [None, [(4, 3)]]
    assert (
        list(s.list_cases_bare(3, degree_sequences_only=True, pure_sequences_only=True))
        == expected
    )

    expected = [
        None,
        [(4, 4)],
        None,
        [(4, 2), (1, 2)],
    ]
    actual = list(
        s.list_cases_bare(4, degree_sequences_only=True, pure_sequences_only=True)
    )

    assert _compare_caselists(actual, expected)

    expected = [
        None,
        [(4, 5)],
        None,
        [(4, 1), (1, 4)],
        None,
        [(1, 2), (4, 3)],
    ]
    actual = list(
        s.list_cases_bare(5, degree_sequences_only=True, pure_sequences_only=True)
    )

    assert _compare_caselists(actual, expected)


def test_empty_selection():
    selection = ncs.Q("")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 3})
        )
        == True
    )


def test_conditional_selection():
    selection = ncs.Q("(C > 4 and H < 2) or (C > 5 and H > 1)")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 5})
        )
        == True
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 6,
                    ncs.AtomType(label="H", valence=4): 2,
                }
            )
        )
        == True
    )


def test_selection_gt():
    selection = ncs.Q("(C > 4)")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == False
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 5})
        )
        == True
    )


def test_selection_ge():
    selection = ncs.Q("(C >= 4)")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 3})
        )
        == False
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == True
    )


def test_selection_le():
    selection = ncs.Q("(C <= 4)")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == True
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 5})
        )
        == False
    )


def test_selection_lt():
    selection = ncs.Q("(C < 4)")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == False
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 3})
        )
        == True
    )


def test_selection_eq():
    selection = ncs.Q("(C = 4)")
    for natoms, result in ((3, False), (4, True), (5, False)):
        assert (
            selection.selected_stoichiometry(
                ncs.AtomStoichiometry(
                    components={ncs.AtomType(label="C", valence=4): natoms}
                )
            )
            == result
        )


def test_selection_eq_multiple():
    selection = ncs.Q("C = 1 & N = 1 & F = 0 & Cl = 0")
    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=4): 1,
            }
        )
    )


def test_selection_word_elements():
    selection = ncs.Q("Carbon < 2")

    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="Carbon", valence=4): 1,
            }
        )
    )
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="Carbon", valence=4): 2,
            }
        )
    )


def test_selection_multiletter_elements():
    selection = ncs.Q("Cl < 2")

    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="Cl", valence=1): 1,
            }
        )
    )
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="Cl", valence=1): 2,
            }
        )
    )


def test_selection_invalid_queries():
    invalids = [
        "C",
        "not C",
        "C# < 2",
        "C | N",
        "2 | 3",
        "C < ()",
        "C < (not C)",
        "C < (N < 3)",
        "C+N < 3 & 4",
        "(C < 3",
    ]
    for invalid in invalids:
        with pytest.raises(ValueError):
            ncs.Q(invalid)


def test_selection_addition():
    selection = ncs.Q("C+N < 3")
    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=3): 1,
            }
        )
    )
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 2,
                ncs.AtomType(label="N", valence=3): 1,
            }
        )
    )


def test_selection_addition_limit():
    # H7F6N23O4
    selection = ncs.Q("C+O+N+F <= 9")
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="H", valence=1): 7,
                ncs.AtomType(label="F", valence=1): 6,
                ncs.AtomType(label="N", valence=3): 23,
                ncs.AtomType(label="O", valence=2): 4,
            }
        )
    )


fuzzydb = [
    (
        "(C > 2) AND (H > 6) AND (N = 1)",
        ({"C": 3, "H": 7, "N": 1}, {"C": 4, "H": 8, "N": 1}, {"C": 5, "H": 10, "N": 1}),
        ({"C": 2, "H": 6, "N": 1}, {"C": 3, "H": 6, "N": 1}, {"C": 3, "H": 7, "N": 2}),
    ),
    (
        "(O > 2) OR (S >= 1) AND (Cl = 0)",
        ({"O": 3}, {"O": 4, "S": 0}, {"S": 1}, {"S": 1, "O": 2}),
        ({"O": 2}, {"S": 1, "Cl": 1}, {"S": 0, "O": 0}),
    ),
    (
        "(C >= 2) AND (H > 4) AND (F > 1)",
        ({"C": 2, "H": 5, "F": 2}, {"C": 3, "H": 6, "F": 2}, {"C": 4, "H": 10, "F": 2}),
        ({"C": 1, "H": 4, "F": 2}, {"C": 2, "H": 5, "F": 1}),
    ),
    (
        "(N > 1) OR ((S = 1) AND (H < 6))",
        ({"N": 2}, {"N": 3}, {"S": 1, "H": 5}, {"S": 1, "H": 4}),
        ({"N": 1}, {"S": 1, "H": 6}, {"S": 2, "H": 3}),
    ),
    (
        "(C = 1) AND ((F >= 1) OR (Cl >= 2))",
        ({"C": 1, "F": 1}, {"C": 1, "Cl": 2}, {"C": 1, "F": 2}),
        ({"C": 2, "F": 1}, {"C": 1, "Cl": 1}, {"C": 1, "F": 0}),
    ),
    (
        "(H >= 4) AND (Cl = 1) AND (F > 0)",
        ({"H": 4, "Cl": 1, "F": 1}, {"H": 5, "Cl": 1, "F": 2}),
        ({"H": 3, "Cl": 1, "F": 1}, {"H": 4, "Cl": 0, "F": 1}),
    ),
    (
        "(C > 1) AND ((O > 1) OR (N >= 1) AND (S = 1))",
        ({"C": 2, "O": 2}, {"C": 3, "S": 1, "N": 1}, {"C": 3, "O": 3}),
        ({"C": 1, "O": 2}, {"C": 2, "O": 1, "S": 1}),
    ),
    (
        "(C = 3) AND (H = 6) AND (Cl >= 2)",
        ({"C": 3, "H": 6, "Cl": 2}, {"C": 3, "H": 6, "Cl": 3}),
        ({"C": 3, "H": 5, "Cl": 2}, {"C": 4, "H": 6, "Cl": 2}),
    ),
    (
        "(C = 0) AND ((O > 2) OR (H > 1) AND (S >= 1))",
        ({"O": 3}, {"H": 2, "S": 1}, {"S": 1, "O": 4}),
        ({"C": 1, "O": 3}, {"H": 1, "S": 1}),
    ),
    (
        "((H > 2) OR (Cl > 1)) AND ((C > 2) AND (O > 1))",
        ({"H": 3, "C": 3, "O": 2}, {"C": 3, "O": 2, "Cl": 2}),
        ({"H": 2, "C": 3, "O": 1}, {"C": 2, "O": 2}),
    ),
    (
        "(C > 3) AND (H <= 6) AND (O = 1)",
        (
            {"C": 4, "H": 6, "O": 1},
            {"C": 5, "H": 4, "O": 1},
            {"C": 6, "H": 5, "O": 1},
            {"C": 4, "H": 3, "O": 1},
            {"C": 5, "H": 6, "O": 1},
        ),
        (
            {"C": 3, "H": 6, "O": 1},
            {"C": 4, "H": 7, "O": 1},
            {"C": 2, "H": 6, "O": 0},
            {"C": 4, "H": 6, "O": 0},
            {"C": 5, "H": 7, "O": 2},
        ),
    ),
    (
        "(C = 2) AND (H = 4) AND (N > 1)",
        (
            {"C": 2, "H": 4, "N": 2},
            {"C": 2, "H": 4, "N": 3},
            {"C": 2, "H": 4, "N": 4},
            {"C": 2, "H": 4, "N": 5},
            {"C": 2, "H": 4, "N": 6},
        ),
        (
            {"C": 2, "H": 5, "N": 2},
            {"C": 3, "H": 4, "N": 2},
            {"C": 1, "H": 4, "N": 1},
            {"C": 2, "H": 4, "N": 1},
            {"C": 2, "H": 3, "N": 2},
        ),
    ),
    (
        "(H >= 6) AND (F = 2) AND (Cl < 3)",
        (
            {"H": 6, "F": 2, "Cl": 2},
            {"H": 7, "F": 2, "Cl": 1},
            {"H": 8, "F": 2, "Cl": 0},
            {"H": 9, "F": 2, "Cl": 1},
            {"H": 6, "F": 2, "Cl": 0},
        ),
        (
            {"H": 5, "F": 2, "Cl": 2},
            {"H": 6, "F": 1, "Cl": 2},
            {"H": 6, "F": 2, "Cl": 3},
            {"H": 4, "F": 2, "Cl": 0},
            {"H": 7, "F": 3, "Cl": 1},
        ),
    ),
    (
        "(C = 1) AND (O > 2) AND (H > 2)",
        (
            {"C": 1, "O": 3, "H": 3},
            {"C": 1, "O": 4, "H": 4},
            {"C": 1, "O": 5, "H": 5},
            {"C": 1, "O": 6, "H": 6},
            {"C": 1, "O": 3, "H": 4},
        ),
        (
            {"C": 1, "O": 2, "H": 3},
            {"C": 1, "O": 1, "H": 4},
            {"C": 2, "O": 3, "H": 3},
            {"C": 1, "O": 3, "H": 2},
            {"C": 0, "O": 4, "H": 3},
        ),
    ),
    (
        "(N >= 2) AND (H = 6) AND (C > 1)",
        (
            {"N": 2, "H": 6, "C": 2},
            {"N": 3, "H": 6, "C": 3},
            {"N": 4, "H": 6, "C": 4},
            {"N": 2, "H": 6, "C": 5},
            {"N": 3, "H": 6, "C": 2},
        ),
        (
            {"N": 1, "H": 6, "C": 2},
            {"N": 2, "H": 5, "C": 2},
            {"N": 2, "H": 6, "C": 1},
            {"N": 0, "H": 6, "C": 3},
            {"N": 3, "H": 7, "C": 2},
        ),
    ),
    (
        "(C > 2) AND ((H = 6) OR (F = 1))",
        (
            {"C": 3, "H": 6},
            {"C": 4, "H": 6},
            {"C": 5, "F": 1},
            {"C": 6, "F": 1},
            {"C": 3, "H": 6, "F": 0},
        ),
        (
            {"C": 2, "H": 6},
            {"C": 1, "F": 1},
            {"C": 3, "H": 5},
            {"C": 4, "H": 7},
            {"C": 2, "H": 6, "F": 1},
        ),
    ),
    (
        "(C > 0) AND (N > 0) AND (H > 4)",
        (
            {"C": 1, "N": 1, "H": 5},
            {"C": 2, "N": 2, "H": 6},
            {"C": 3, "N": 1, "H": 7},
            {"C": 4, "N": 3, "H": 8},
            {"C": 1, "N": 2, "H": 5},
        ),
        (
            {"C": 0, "N": 1, "H": 5},
            {"C": 1, "N": 0, "H": 5},
            {"C": 1, "N": 1, "H": 4},
            {"C": 2, "N": 0, "H": 3},
            {"C": 1, "N": 1, "H": 3},
        ),
    ),
    (
        "(O > 2) OR (S = 1) OR (Cl > 1)",
        ({"O": 3}, {"S": 1}, {"Cl": 2}, {"O": 4, "S": 1}, {"Cl": 3, "O": 0}),
        ({"O": 2}, {"S": 0}, {"Cl": 1}, {"O": 1, "S": 0, "Cl": 0}, {"O": 0, "Cl": 0}),
    ),
    (
        "(C = 2) AND (H > 2) AND (F <= 1)",
        (
            {"C": 2, "H": 3, "F": 1},
            {"C": 2, "H": 4, "F": 0},
            {"C": 2, "H": 5, "F": 1},
            {"C": 2, "H": 6, "F": 0},
            {"C": 2, "H": 7, "F": 1},
        ),
        (
            {"C": 2, "H": 2, "F": 1},
            {"C": 2, "H": 3, "F": 2},
            {"C": 1, "H": 3, "F": 1},
            {"C": 3, "H": 4, "F": 0},
            {"C": 2, "H": 3, "F": 2},
        ),
    ),
    (
        "(C > 3) AND (H > 5) AND (O >= 2)",
        (
            {"C": 4, "H": 6, "O": 2},
            {"C": 5, "H": 8, "O": 3},
            {"C": 6, "H": 7, "O": 4},
            {"C": 4, "H": 6, "O": 2},
            {"C": 5, "H": 9, "O": 3},
        ),
        (
            {"C": 3, "H": 6, "O": 2},
            {"C": 4, "H": 5, "O": 2},
            {"C": 4, "H": 6, "O": 1},
            {"C": 2, "H": 6, "O": 2},
            {"C": 4, "H": 4, "O": 2},
        ),
    ),
]


def _matching_to_components(matching: str):
    valences = {"H": 1, "C": 4, "N": 3, "O": 2, "Cl": 1, "S": 2, "F": 1}
    components = {}
    for label, count in matching.items():
        components[ncs.AtomType(label=label, valence=valences[label])] = count
    return components


@pytest.mark.parametrize(
    "selection,matching", [(_[0], __) for _ in fuzzydb for __ in _[1]]
)
def test_cases_fuzzy_matching(selection: str, matching: str):
    selection = ncs.Q(selection)
    components = _matching_to_components(matching)

    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(components=components)
    )


@pytest.mark.parametrize(
    "selection,matching", [(_[0], __) for _ in fuzzydb for __ in _[2]]
)
def test_cases_fuzzy_nonmatching(selection: str, matching: str):
    selection = ncs.Q(selection)
    components = _matching_to_components(matching)
    result = selection.selected_stoichiometry(
        ncs.AtomStoichiometry(components=components)
    )
    assert not result


def test_selection_addition_multiple():
    selection = ncs.Q("C+N+O < 3")
    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=3): 1,
            }
        )
    )
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 2,
                ncs.AtomType(label="N", valence=3): 1,
            }
        )
    )


def test_selection_two_additions():
    selection = ncs.Q("C+N < N+F")
    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=3): 0,
                ncs.AtomType(label="F", valence=1): 2,
            }
        )
    )
    assert not selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=3): 0,
                ncs.AtomType(label="F", valence=1): 1,
            }
        )
    )


def test_selection_multiple_andor_precedence():
    for case in ("C = 1 & N = 1 | F = 0", "F = 0 | C = 1 & N = 1"):
        selection = ncs.Q(case)
        assert selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 2,
                    ncs.AtomType(label="N", valence=4): 1,
                }
            )
        )
        assert not selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 1,
                    ncs.AtomType(label="F", valence=4): 1,
                }
            )
        )


def test_counter_value(approximate_counter):
    s = ncs.SearchSpace.covered_search_space("B")
    q = ncs.Q("C < 4 & H > 2")
    assert approximate_counter.count(s, 9) == 180869284
    assert approximate_counter.count(s, 10, q) == 7333401

    s = ncs.SearchSpace.covered_search_space("A")
    assert approximate_counter.count(s, 9) == 178099319914
    assert approximate_counter.count(s, 10, q) == 1486411142


def test_count_one_paths(approximate_counter):
    ## pure
    # exact in database
    label = (1, 2)
    assert label in approximate_counter._exact_db
    assert approximate_counter.count_one_bare(label, 2) == 1

    # avg path length in db, but not exact
    ## count = exp(1.220*lg - 0.7295)
    ## lg = 55.0 for 6.20
    label = (6, 20)
    assert label not in approximate_counter._exact_db
    assert label in approximate_counter._approx_db
    assert approximate_counter.count_one_bare(label, 20) == np.exp(
        1.22066271 * 55.0 - 0.72953204
    )

    # avg path length not in db, use asymptotics
    # 6.21 not in db
    ## M = 6*21 = 126
    ## M_2 = 30*21 = 630
    ## M_3 = 120*21 = 2520
    ## y1 = 0
    ## x2 = 1
    ## x3 = 1
    ## prefactor = M! / ((M/2)! * 2**(M/2) * (6!)**21)
    ##           = 23721732428800468856771473051394170805702085973808045661837377170052497697783313457227249544076486314839447086187187275319400401837013955325179315652376928996065123321190898603130880000000000000000000000000000000 /
    ##             (1982608315404440064116146708361898137544773690227268628106279599612729753600000000000000 * 9223372036854775808 * 1009212044656507725162109374628859215872000000000000000000000)
    ##           = 2752775426286713065002722116483278236963996020922703261154808938845230629735001 / 2141575821672727144116749598720000 ~ 1.2853971353377506e+45
    ## term1 = (y1-1/2) * M_2/M = (0-1/2) * 630/126 = -5 / 2
    ## term2 = (x2-1/2) * M_2**2 / (2*M**2) = (1-1/2) * 630**2 / (2*126**2) = 25 / 4
    ## term3 = M_2**4 / (4*M**5)  = 630**4 / (4*126**5) = 625 / 504
    ## term4 = -M_2**2*M_3/ (2*M**4) = -630**2*2520/ (2*126**4) =-125/63
    ## term5 = (x3-x2+1/3) * M_3**2/(2*M**3) = 1/3 * 2520**2/(2*126**3) = 100/189
    ## term1+...+term5 = -5 / 2 + 25 / 4 + 625 / 504 - 125/63 + 100/189 = 5345/1512 ~ 3.53505291005291
    ## G = 1.2853971353377506e+45 * np.exp(3.53505291005291) = 4.408504615045374e+46
    ## lG = 0.7561 * log(G) - 14.4 = 66.81448818253898
    ## count = exp(1.220*lg - 0.7295) = 1.2686380422711015e+35
    label = (6, 21)
    assert label not in approximate_counter._exact_db
    assert label not in approximate_counter._approx_db
    assert approximate_counter.count_one_bare(label, 21) == pytest.approx(
        1.2686380422711015e35
    )

    ## non-pure
    # exact in database
    label = (4, 1, 4, 8)
    assert label in approximate_counter._exact_db
    assert approximate_counter.count_one_bare(label, 2) == 5463

    # avg path length in db
    ## count = exp(1.220*lg - 0.7295)
    ## lg = 32.4 for (3, 1, 3, 1, 5, 14)
    label = (3, 1, 3, 1, 5, 14)
    assert label not in approximate_counter._exact_db
    assert label in approximate_counter._approx_db
    assert approximate_counter.count_one_bare(label, 16) == np.exp(
        1.22066271 * 32.4 - 0.72953204
    )

    # only pure avg path length in db but non-pure avg path not in db
    label = (6, 2, 6, 8, 6, 10)
    pure_label = (6, 20)
    assert label not in approximate_counter._exact_db
    assert label not in approximate_counter._approx_db
    assert pure_label not in approximate_counter._exact_db
    assert pure_label in approximate_counter._approx_db
    ## Np = (2+8+10 choose 2) * (8+10 choose 8) * (10 choose 10) = 190 * 43758 * 1 = 8314020
    ## M = (20*6) = 120
    ## prefactor = log(Np) / M +1 = log(8314020) / 120 +1= 1.13277878170310645
    ## lg_pure for (6, 20) is 55.0
    ## lg_nonpure = prefactor * lg_pure = 1.13277878170310645 * 55.0 = 62.302832
    ## count = exp(1.220*lg_nonpure - 0.7295)
    assert approximate_counter.count_one_bare(label, 20) == pytest.approx(
        np.exp(1.22066271 * 62.302832 - 0.72953204), 1e-4
    )

    # avg path length not in db, also pure avg path not in db use asymptotics
    label = (6, 19, 6, 2)
    pure_label = (6, 21)
    assert label not in approximate_counter._exact_db
    assert label not in approximate_counter._approx_db
    assert pure_label not in approximate_counter._exact_db
    assert pure_label not in approximate_counter._approx_db
    # same case as above, only non-pure prefactor needs checking
    ## Np = (19+2 choose 19) * (2 choose 2) = 210 * 1
    ## M = (21*6) = 126
    ## prefactor = log(Np) / M + 1 = log(210) / 126 + 1 = 1.042437361354900
    ## lg_pure for (6, 21) is 66.81448818253898 (see above)
    ## lg_nonpure = prefactor * lg_pure = 1.042437361354900 * 66.81448818253898 = 69.64991876128408
    ## count = exp(1.220*lg_nonpure - 0.7295)
    assert approximate_counter.count_one_bare(label, 21) == pytest.approx(
        4.040882863399748e36
    )


def test_inverse_path_length(approximate_counter):
    lg = 10
    size = approximate_counter._average_path_length_to_size(lg)
    lgnew = approximate_counter._size_to_average_path_length(size)
    assert lg == pytest.approx(lgnew, rel=1e-5)


def test_inverse_path_length_big_number(approximate_counter):
    size = 10000000000000000000000
    approximate_counter._size_to_average_path_length(size)


def test_min_zero_count(approximate_counter):
    assert approximate_counter._average_path_length_to_size(-1) == 0


def test_selection_eq_multiple_mixed_operators():
    selection = ncs.Q("C = 1 and N = 1 & F = 0 and Cl = 0")
    assert selection.selected_stoichiometry(
        ncs.AtomStoichiometry(
            components={
                ncs.AtomType(label="C", valence=4): 1,
                ncs.AtomType(label="N", valence=4): 1,
            }
        )
    )


def test_pure_permutation_prefactor(approximate_counter):
    groups = (2, 3, 4)
    # (2+3+4 choose 2) * (3+4 choose 3) * (4 choose 4) = 36 * 35 * 1 = 1260
    assert approximate_counter._cached_permutation_factor_log(groups) == np.log(1260)
    groups = (2,)
    assert approximate_counter._cached_permutation_factor_log(groups) == np.log(1)


def test_count_by_pure_lookup(approximate_counter):
    case_for_which_pure_entry_exists = (1, 1, 1, 14, 3, 1, 4, 7)
    assert (
        approximate_counter.count_one_bare(case_for_which_pure_entry_exists, 23) == 801
    )


def test_selection_neq():
    selection = ncs.Q("(C != 4)")
    for natoms, result in ((3, True), (4, False), (5, True)):
        assert (
            selection.selected_stoichiometry(
                ncs.AtomStoichiometry(
                    components={ncs.AtomType(label="C", valence=4): natoms}
                )
            )
            == result
        )


def test_selection_and():
    for op in ("and", "&"):
        selection = ncs.Q(f"(C > 2) {op} (C < 5)")
        for natoms, result in ((2, False), (3, True), (4, True), (5, False)):
            assert (
                selection.selected_stoichiometry(
                    ncs.AtomStoichiometry(
                        components={ncs.AtomType(label="C", valence=4): natoms}
                    )
                )
                == result
            )


def test_selection_or():
    for op in ("or", "|"):
        selection = ncs.Q(f"(C > 3) {op} (C < 3)")
        for natoms, result in ((2, True), (3, False), (4, True)):
            assert (
                selection.selected_stoichiometry(
                    ncs.AtomStoichiometry(
                        components={ncs.AtomType(label="C", valence=4): natoms}
                    )
                )
                == result
            )


def test_selection_not():
    for op in ("not", "!"):
        selection = ncs.Q(f"{op}(C > 3)")
        for natoms, result in ((2, True), (3, True), (4, False)):
            assert (
                selection.selected_stoichiometry(
                    ncs.AtomStoichiometry(
                        components={ncs.AtomType(label="C", valence=4): natoms}
                    )
                )
                == result
            )


def test_selection_exclude_element():
    selection = ncs.Q("He =0")
    for natoms, result in ((0, True), (1, False)):
        assert (
            selection.selected_stoichiometry(
                ncs.AtomStoichiometry(
                    components={ncs.AtomType(label="He", valence=4): natoms}
                )
            )
            == result
        )


def test_limit_natoms():
    selection = ncs.Q("# < 4")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == False
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 3})
        )
        == True
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 2,
                    ncs.AtomType(label="H", valence=4): 1,
                }
            )
        )
        == True
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 2,
                    ncs.AtomType(label="H", valence=4): 2,
                }
            )
        )
        == False
    )


def test_limit_inverse():
    selection = ncs.Q("4 < C")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 4})
        )
        == False
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(components={ncs.AtomType(label="C", valence=4): 5})
        )
        == True
    )


def test_limit_relation():
    selection = ncs.Q("H < C")
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 2,
                    ncs.AtomType(label="H", valence=4): 1,
                }
            )
        )
        == True
    )
    assert (
        selection.selected_stoichiometry(
            ncs.AtomStoichiometry(
                components={
                    ncs.AtomType(label="C", valence=4): 2,
                    ncs.AtomType(label="H", valence=4): 2,
                }
            )
        )
        == False
    )


@pytest.mark.parametrize(
    "letter,natoms", [(letter, natoms) for letter in "AB" for natoms in range(3, 30)]
)
def test_sum_formula_database_covered(letter, natoms, approximate_counter):
    s = ncs.SearchSpace.covered_search_space(letter)
    switchover = {"A": 15, "B": 22}
    assert (
        approximate_counter.missing_parameters(s, natoms, natoms > switchover[letter])
        == []
    )


def test_sum_formula_countable(approximate_counter):
    s = ncs.SearchSpace.covered_search_space("B")
    for i in range(3, 10):
        approximate_counter.count(s, i)


@pytest.mark.timeout(2)
def test_filterlist():
    space = ncs.SearchSpace("C:4 H:1 N:3 O:2 F:1 Cl:1 Br:1 S:2,4,6 P:3,5")
    counter = ncs.ExactCounter("/bin/false")  # should never be called anyway
    assert list(counter.list(space, natoms=8, selection=ncs.Q("# = 0"))) == []


def test_issue4(approximate_counter):
    space = ncs.SearchSpace("C:4 S:2,4")
    approximate_counter.count(space, natoms=20)


def test_small(approximate_counter):
    space = ncs.SearchSpace("H:1")
    assert approximate_counter.count(space, natoms=2) > 0


def test_ring(approximate_counter):
    space = ncs.SearchSpace("O:2")
    for natoms in range(2, 10):
        assert approximate_counter.count(space, natoms=natoms) > 0


def test_only_one_partition(approximate_counter):
    space = ncs.SearchSpace("S:4")
    assert approximate_counter.count(space, natoms=4) > 0


def test_shortcut_surge_hydrogen():
    counter = ncs.ExactCounter("/bin/false")

    # shortcut of using built-in special treatment of Hydrogens
    stoichiometry = ncs.AtomStoichiometry(
        components={ncs.AtomType("C", 4): 9, ncs.AtomType("F", 1): 20}
    )
    args, mapping = counter._build_cli_arguments(stoichiometry, count_only=True)
    assert "-c4 -d4 -EAa44 -u Aa9H20" == args
    assert mapping["H"] == "F"

    # only use the shortcut if this is about counting
    args, mapping = counter._build_cli_arguments(stoichiometry, count_only=False)
    assert "-c4 -d4 -EAa44 -EAb11 -A Aa9Ab20" == args
    assert mapping["Ab"] == "F"
