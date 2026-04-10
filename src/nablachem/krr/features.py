import numpy as np
from . import dataset
import inspect


class BaseRepresenter:
    def _prepare(self, molecules: list) -> None:
        pass

    def compute(self, molecules: list) -> list:
        raise NotImplementedError

    def build(self, ds, compatible_to=None) -> None:
        other_mols = [m for other in (compatible_to or []) for m in other.molecules]
        self._molecules = ds.molecules
        self._prepare(ds.molecules + other_mols)
        ds.representations = self

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.compute(self._molecules[key])
        return self.compute([self._molecules[key]])[0]


class _DF(BaseRepresenter):
    def __init__(self, local: bool, dep: str):
        self._local = local
        self._dep = dep

    def compute(self, molecules: list) -> list:
        if self._dep == "mbdf":
            from .deps import mbdf
            call = mbdf.generate_mbdf
        else:
            from .deps import cmbdf
            call = cmbdf.generate_mbdf

        mols_charges = [mol.get_atomic_numbers() for mol in molecules]
        mols_coords = [mol.get_positions() for mol in molecules]
        natoms = [len(c) for c in mols_charges]

        reps = call(mols_charges, mols_coords, progress_bar=False, local=self._local)

        if self._local:
            return [reps[i][:natoms[i], :] for i in range(len(molecules))]
        else:
            return list(reps)


class MBDFLocal(_DF):
    def __init__(self):
        super().__init__(local=True, dep="mbdf")


class MBDFGlobal(_DF):
    def __init__(self):
        super().__init__(local=False, dep="mbdf")


class cMBDFLocal(_DF):
    def __init__(self):
        super().__init__(local=True, dep="cmbdf")


class cMBDFGlobal(_DF):
    def __init__(self):
        super().__init__(local=False, dep="cmbdf")


class _SLATM(BaseRepresenter):
    def __init__(self, local: bool = False):
        self._local = local

    def _prepare(self, molecules: list) -> None:
        import qmllib.representations
        all_charges = [mol.get_atomic_numbers() for mol in molecules]
        self._mbtypes = qmllib.representations.get_slatm_mbtypes(all_charges)

    def compute(self, molecules: list) -> list:
        import qmllib.representations
        reps = []
        for mol in molecules:
            charges = mol.get_atomic_numbers()
            coords = mol.get_positions()
            rep = qmllib.representations.generate_slatm(
                nuclear_charges=charges,
                coordinates=coords,
                mbtypes=self._mbtypes,
                local=self._local,
            )
            natom = len(charges)
            if self._local:
                if isinstance(rep, list):
                    reps.append(np.array(rep[:natom]))
                else:
                    reps.append(rep[:natom, :])
            else:
                reps.append(rep)
        return reps


class SLATMLocal(_SLATM):
    def __init__(self):
        super().__init__(local=True)


class SLATMGlobal(_SLATM):
    def __init__(self):
        super().__init__(local=False)


class _MACE(BaseRepresenter):
    def __init__(self, local: bool):
        self._local = local
        self._model = None

    def _prepare(self, molecules: list) -> None:
        if self._model is None:
            import warnings
            import contextlib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from mace.calculators import mace_mp
                with contextlib.redirect_stdout(None):
                    self._model = mace_mp(
                        model="medium", device="", default_dtype="float64"
                    )

    def compute(self, molecules: list) -> list:
        if self._local:
            return [self._model.get_descriptors(mol) for mol in molecules]
        else:
            return [np.sum(self._model.get_descriptors(mol), axis=0) for mol in molecules]


class MACEGlobal(_MACE):
    def __init__(self):
        super().__init__(local=False)


class MACELocal(_MACE):
    def __init__(self):
        super().__init__(local=True)


class _FCHL19(BaseRepresenter):
    def __init__(self, local: bool):
        self._local = local

    def _prepare(self, molecules: list) -> None:
        self._all_element_numbers = sorted(
            set(n for mol in molecules for n in mol.get_atomic_numbers())
        )

    def compute(self, molecules: list) -> list:
        import qmllib.representations
        reps = []
        for mol in molecules:
            rep = qmllib.representations.generate_fchl19(
                mol.get_atomic_numbers(),
                mol.get_positions(),
                elements=self._all_element_numbers,
            )
            if not self._local:
                rep = np.sum(rep, axis=0)
            reps.append(rep)
        return reps


class FCHL19Global(_FCHL19):
    def __init__(self):
        super().__init__(local=False)


class FCHL19Local(_FCHL19):
    def __init__(self):
        super().__init__(local=True)


def list_available():
    """Return string names of all BaseRepresenter subclasses that don't start with underscore."""
    current_module = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isclass
    )

    available = []
    for name, cls in current_module:
        if (
            name != "BaseRepresenter"
            and issubclass(cls, BaseRepresenter)
            and not name.startswith("_")
        ):
            available.append(name)

    return available
