import nablachem.utils as ncu
import pytest


def test_atomtype_label_simple():
    with pytest.raises(ValueError):
        ncu.AtomType(label="something with space", valence=1)

    with pytest.raises(ValueError):
        ncu.AtomType(label="with<special>characters", valence=1)
