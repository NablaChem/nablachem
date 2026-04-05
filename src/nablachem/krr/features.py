import numpy as np
from . import dataset
import inspect


class BaseRepresenter:
    def build(self, datasets: list[dataset.DataSet]) -> None: ...


class _DF(BaseRepresenter):
    def __init__(self, local: bool, dep: str):
        self._local = local
        self._dep = dep

    def build(self, datasets: list[dataset.DataSet]):
        if self._dep == "mbdf":
            from .deps import mbdf
            call = mbdf.generate_mbdf
        else:
            from .deps import cmbdf
            call = cmbdf.generate_mbdf

        mols_charges = []
        mols_coords = []
        natoms = []
        for ds in datasets:
            for mol in ds.molecules:
                mols_charges.append(mol.get_atomic_numbers())
                mols_coords.append(mol.get_positions())
                natoms.append(len(mol.get_atomic_numbers()))
        reps = call(
            mols_charges, mols_coords, progress_bar=False, local=self._local
        )

        if self._local:
            reps_short = []
            for idx, natom in enumerate(natoms):
                reps_short.append(reps[idx][:natom, :])
        else:
            reps_short = [rep for rep in reps]

        offset = 0
        for ds in datasets:
            ds.representations = reps_short[offset : offset + len(ds.molecules)]
            offset += len(ds.molecules)


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

    def build(self, datasets: list[dataset.DataSet]):
        # Collect nuclear charges and coordinates from all molecules
        all_nuclear_charges = []
        mols_charges = []
        mols_coords = []
        natoms = []

        for ds in datasets:
            for mol in ds.molecules:
                charges = mol.get_atomic_numbers()
                coords = mol.get_positions()
                all_nuclear_charges.append(charges)
                mols_charges.append(charges)
                mols_coords.append(coords)
                natoms.append(len(charges))

        import qmllib.representations
        # Get mbtypes for the entire dataset
        mbtypes = qmllib.representations.get_slatm_mbtypes(all_nuclear_charges)

        # Generate SLATM representations for each molecule
        reps = []
        for charges, coords in zip(mols_charges, mols_coords):
            rep = qmllib.representations.generate_slatm(
                nuclear_charges=charges,
                coordinates=coords,
                mbtypes=mbtypes,
                local=self._local,
            )
            reps.append(rep)

        if self._local:
            # For local representation, truncate to actual number of atoms
            reps_short = []
            for idx, natom in enumerate(natoms):
                if isinstance(reps[idx], list):
                    reps_short.append(np.array(reps[idx][:natom]))
                else:
                    reps_short.append(reps[idx][:natom, :])
        else:
            reps_short = [rep for rep in reps]

        # Assign representations to datasets
        offset = 0
        for ds in datasets:
            ds.representations = reps_short[offset : offset + len(ds.molecules)]
            offset += len(ds.molecules)


class SLATMLocal(_SLATM):
    def __init__(self):
        super().__init__(local=True)


class SLATMGlobal(_SLATM):
    def __init__(self):
        super().__init__(local=False)


class _MACE(BaseRepresenter):
    def __init__(self, local: bool, model_path: str = "medium"):
        self._local = local
        self._model_path = model_path
        self._model = None

    def build(self, datasets: list[dataset.DataSet]):
        if self._model is None:
            import warnings
            import contextlib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from mace.calculators import mace_mp

                with contextlib.redirect_stdout(None):
                    self._model = mace_mp(
                        model=self._model_path, device="", default_dtype="float64"
                    )

        all_mols = [mol for ds in datasets for mol in ds.molecules]
        if self._local:
            reps = [self._model.get_descriptors(mol) for mol in all_mols]
        else:
            reps = [
                np.sum(self._model.get_descriptors(mol), axis=0) for mol in all_mols
            ]

        offset = 0
        for ds in datasets:
            ds.representations = reps[offset : offset + len(ds.molecules)]
            offset += len(ds.molecules)


class MACEGlobal(_MACE):
    def __init__(self, model_path: str = "medium"):
        super().__init__(local=False, model_path=model_path)


class MACELocal(_MACE):
    def __init__(self, model_path: str = "medium"):
        super().__init__(local=True, model_path=model_path)


class _FCHL19(BaseRepresenter):
    def __init__(self, local: bool):
        self._local = local

    def build(self, datasets: list[dataset.DataSet]):
        all_mols = [mol for ds in datasets for mol in ds.molecules]
        all_element_numbers = set()
        for mol in all_mols:
            all_element_numbers.update(mol.get_atomic_numbers())
        import qmllib.representations
        all_element_numbers = sorted(all_element_numbers)
        reps = []
        for mol in all_mols:
            rep = qmllib.representations.generate_fchl19(
                mol.get_atomic_numbers(),
                mol.get_positions(),
                elements=all_element_numbers,
            )
            if not self._local:
                rep = np.sum(rep, axis=0)
            reps.append(rep)

        offset = 0
        for ds in datasets:
            ds.representations = reps[offset : offset + len(ds.molecules)]
            offset += len(ds.molecules)


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
