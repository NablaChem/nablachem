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
    for op in ("not", "no", "!"):
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
    selection = ncs.Q("no He")
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
