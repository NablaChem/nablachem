import pyparsing
import pytest
import nablachem.space as ncs


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


def test_counter_value():
    c = ncs.ApproximateCounter()
    s = ncs.SearchSpace.covered_search_space("B")
    q = ncs.Q("C < 4 & H > 2")

    assert c.count(s, 10) == 3754609422
    assert c.count(s, 10, q) == 7333401

    s = ncs.SearchSpace.covered_search_space("A")
    assert c.count(s, 10) == 11608588574694
    assert c.count(s, 10, q) == 1486411142


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
def test_sum_formula_database_covered(letter, natoms):
    s = ncs.SearchSpace.covered_search_space(letter)
    counter = ncs.ApproximateCounter()
    switchover = {"A": 15, "B": 22}
    assert counter.missing_parameters(s, natoms, natoms > switchover[letter]) == []


def test_sum_formula_countable():
    s = ncs.SearchSpace.covered_search_space("B")
    c = ncs.ApproximateCounter()
    for i in range(3, 10):
        c.count(s, i)


@pytest.mark.timeout(2)
def test_big():
    space = ncs.SearchSpace("C:4 H:1 N:3 O:2 F:1 Cl:1 Br:1 S:2,4,6 P:3,5")
    counter = ncs.ExactCounter("/bin/false")  # should never be called anyway
    assert list(counter.list(space, natoms=8, selection=ncs.Q("# = 0"))) == []
