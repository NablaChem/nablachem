import numpy as np
import ase
import ase.io
import gzip
import json
import random
from typing import Union, Callable
from io import StringIO

from utils import info, debug, warning, error


class DataSet:
    def __init__(
        self,
        filename: str,
        labelname: Union[str, Callable[[dict], float]],
        limit: int,
        baseline: str = None,
    ):
        """Read gzipped JSONL file.

        Args:
            filename: Path to .gz file containing JSON lines
            labelname: Property name to extract as labels, or callable that takes a data row and returns a numeric value
                      Examples:
                        - String: "energy" (extracts data["energy"])
                        - Callable: lambda row: row["E_high"] - row["E_low"] (computes difference)
            limit: Maximum number of molecules to load (randomly selected)
            baseline: Optional baseline column name. If provided, labels will be labelname - baseline
        """
        with gzip.open(filename, "rt") as f:
            all_lines = f.readlines()

        if limit is not None and len(all_lines) > limit:
            selected_lines = random.sample(all_lines, limit)
        else:
            selected_lines = all_lines

        molecules = []
        raw_labels = []

        # make everything a callable
        if isinstance(labelname, str):
            if baseline is not None:
                extractor = lambda row: row[labelname] - row[baseline]
            else:
                extractor = lambda row: row[labelname]
        else:
            if baseline is not None:
                extractor = lambda row: labelname(row) - row[baseline]
            else:
                extractor = labelname

        first = True
        for line_idx, line in enumerate(selected_lines):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                error("JSON decode error", line_idx=line_idx, error_msg=str(e))
            if first:
                found_keys = [key for key in data.keys() if key != "xyz"]
                info(
                    "First molecule keys", keys=found_keys, total_keys=len(data.keys())
                )
                first = False

            try:
                molecules.append(ase.io.read(StringIO(data["xyz"]), format="xyz"))
            except Exception as e:
                error(
                    "Failed to parse XYZ for molecule",
                    line_idx=line_idx,
                    error_msg=str(e),
                )

            try:
                label_value = extractor(data)
                raw_labels.append(float(label_value))
            except Exception as e:
                error(
                    "Failed to extract label from molecule",
                    line_idx=line_idx,
                )

        raw_labels = np.array(raw_labels)

        self.molecules = molecules
        self.labels = raw_labels

        # shuffle
        indices = np.arange(len(molecules))
        np.random.shuffle(indices)
        self.molecules = [self.molecules[i] for i in indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.molecules)

    def get_element_counts(self):
        """Return element count matrix for all molecules in the dataset.

        Returns:
            np.ndarray: Matrix of shape (N, k) where N is the number of molecules
                       and k is the number of unique elements. Each entry (i, j)
                       contains the count of element j in molecule i.
        """
        if not self.molecules:
            return np.array([]).reshape(0, 0)

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

        return element_counts

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
