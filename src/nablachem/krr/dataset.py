import numpy as np
from numpy.char import isdigit
import pandas as pd
import ase
import ase.io
from io import StringIO

from .utils import info, warning, error


class DataSet:
    def __init__(
        self,
        filename: str,
        labelname: str,
        limit: int = None,
        select: str = None,
    ):
        """Read gzipped JSONL file.

        Args:
            filename: Path to .gz file containing JSON lines
            labelname: String expression for pandas DataFrame.eval() to compute labels
                      Examples: "energy", "energy - baseline", "E_high - E_low"
            limit: Maximum number of molecules to load (None = no limit)
            select: Optional selection expression for pandas DataFrame.query()
        """
        try:
            df = pd.read_json(filename, lines=True)
        except Exception as e:
            error("Failed to load JSONL file", filename=filename, error_msg=str(e))

        if "xyz" not in df.columns:
            error(
                "Required 'xyz' column not found in dataset",
                columns=df.columns.tolist(),
            )

        atom_cols = pd.DataFrame(df["xyz"].apply(self._parse_xyz_counts).tolist())
        atom_cols = atom_cols.fillna(0).astype(int)

        df = pd.concat([df, atom_cols], axis=1)

        found_keys = [col for col in df.columns if col != "xyz"]
        info(
            "Dataset columns",
            columns=found_keys,
            total_columns=len(df.columns),
            total_rows=len(df),
        )

        if select is not None:
            try:
                starting_rows = len(df)
                df = df.query(select)
                remaining_rows = len(df)
                if remaining_rows == starting_rows:
                    warning("Selection without effect", select=select)
                elif remaining_rows == 0:
                    error(
                        "There are no remaining rows", filename=filename, select=select
                    )
                else:
                    info(
                        "Applied selection",
                        select=select,
                        remaining_rows=remaining_rows,
                    )

            except Exception as e:
                error("Failed to apply selection", select=select, error_msg=str(e))

        df = df.sample(frac=1).reset_index(drop=True)

        if limit is not None:
            df = df.head(limit)
            info("Applied limit", limit=limit, final_rows=len(df))

        try:
            labels = df.eval(labelname)
            self.labels = np.array(labels, dtype=float)
            info(
                "Computed labels",
                labelname=labelname,
                sample_labels=self.labels[:5].tolist(),
            )
        except Exception as e:
            error(
                "Failed to evaluate labelname expression",
                labelname=labelname,
                error_msg=str(e),
            )
            raise

        molecules = []
        for idx, xyz_data in enumerate(df["xyz"]):
            try:
                molecules.append(ase.io.read(StringIO(xyz_data), format="xyz"))
            except Exception as e:
                error(
                    "Failed to parse XYZ for molecule",
                    molecule_idx=idx,
                    error_msg=str(e),
                )

        self.molecules = molecules
        del df

    def __len__(self):
        return len(self.molecules)

    def get_element_counts(self):
        """Return element count matrix for all molecules in the dataset.

        Returns:
            np.ndarray: Matrix of shape (N, k) where N is the number of molecules
                       and k is the number of unique elements. Each entry (i, j)
                       contains the count of element j in molecule i.
            list[int]: List of unique atomic numbers corresponding to columns
                       in the element count matrix.
        """
        if not self.molecules:
            return np.array([]).reshape(0, 0), []

        # Get all unique atomic numbers across all molecules
        all_atomic_numbers = set()
        for mol in self.molecules:
            all_atomic_numbers.update(mol.get_atomic_numbers())

        # Sort to ensure consistent ordering
        unique_atomic_numbers = sorted(all_atomic_numbers)

        # Create element count matrix
        element_counts = np.zeros((len(self.molecules), len(unique_atomic_numbers)))

        for mol_idx, mol in enumerate(self.molecules):
            atomic_numbers = mol.get_atomic_numbers()
            for element_idx, atomic_num in enumerate(unique_atomic_numbers):
                element_counts[mol_idx, element_idx] = np.sum(
                    atomic_numbers == atomic_num
                )

        return element_counts, unique_atomic_numbers

    @staticmethod
    def _parse_xyz_counts(xyz: str) -> dict:
        import ase.data

        lines = xyz.split("\n")
        number_atoms = int(lines[0].strip())
        counts = {"n_atoms": number_atoms}

        for line in lines[2 : 2 + number_atoms]:
            atom = line.split()[0]

            if atom.isdigit():
                symbol = ase.data.chemical_symbols[int(atom)]
            else:
                symbol = atom

            counts[f"n_{symbol}"] = counts.get(f"n_{symbol}", 0) + 1
        return counts

    def write_holdout_residuals_jsonl(
        self,
        holdout_residuals: dict[int, np.ndarray],
        holdout_start_idx: int,
        output_path: str,
    ) -> None:
        """Write holdout molecules with residuals to JSONL file.

        Args:
            holdout_residuals: Dict mapping training size to residual arrays
            holdout_start_idx: Index where holdout data starts in the dataset
            output_path: Path for output JSONL file
        """
        import gzip
        import json
        from io import StringIO

        # Skip nullmodel (training size 1)
        training_sizes = sorted([k for k in holdout_residuals.keys() if k > 1])

        with gzip.open(output_path, "wt") as f:
            for i, mol in enumerate(self.molecules[holdout_start_idx:]):
                # Convert molecule to xyz string
                xyz_buffer = StringIO()
                ase.io.write(xyz_buffer, mol, format="xyz")
                xyz_string = xyz_buffer.getvalue().strip()

                # Create output record with xyz and residual columns
                record = {"xyz": xyz_string}
                for ntrain in training_sizes:
                    record[f"N{ntrain}"] = float(holdout_residuals[ntrain][i])

                f.write(json.dumps(record) + "\n")
