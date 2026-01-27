import ase
import numpy as np
import cmbdf
import dataset
import qmllib.representations


class BaseRepresenter:
    def build(self, datasets: list[dataset.DataSet]) -> None: ...

    def exponential_kernel(
        self, X: list[np.ndarray], Y: list[np.ndarray], gamma: float
    ) -> np.ndarray: ...


class _cMBDF(BaseRepresenter):
    def __init__(self, local: bool = False):
        self._local = local

    def build(self, datasets: list[dataset.DataSet]):
        mols_charges = []
        mols_coords = []
        natoms = []
        for ds in datasets:
            for mol in ds.molecules:
                mols_charges.append(mol.get_atomic_numbers())
                mols_coords.append(mol.get_positions())
                natoms.append(len(mol.get_atomic_numbers()))
        reps = cmbdf.generate_mbdf(
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


class cMBDFLocal(_cMBDF):
    def __init__(self):
        super().__init__(local=True)


class cMBDFGlobal(_cMBDF):
    def __init__(self):
        super().__init__(local=False)


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
                    reps_short.append(reps[idx][:natom])
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
