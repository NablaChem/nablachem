import gzip
import random
import re
from io import StringIO

import ase
import ase.io
import numpy as np
import pandas as pd

from .utils import info, warning, error

_CALC_COL_RE = re.compile(r"\bn_atoms\b|\bn_[A-Z][a-z]?\b")


class DataSet:
    def __init__(
        self,
        filename: str,
        labelname: str,
        limit: int = None,
        select: str = None,
    ):
        """Read gzipped or plain JSONL file.

        Args:
            filename: Path to .jsonl or .jsonl.gz file containing JSON lines
            labelname: String expression for pandas DataFrame.eval() to compute labels
                      Examples: "energy", "energy - baseline", "E_high - E_low"
            limit: Maximum number of molecules to load (None = no limit)
            select: Optional selection expression for pandas DataFrame.query()
        """
        if limit is not None:
            df = DataSet._reservoir_sample(filename, limit, select)
        else:
            try:
                df = pd.read_json(filename, lines=True)
            except Exception as e:
                error("Failed to load JSONL file", filename=filename, error_msg=str(e))

            mode = DataSet._detect_select_mode(select)

            if select is not None:
                if mode == "C":
                    atom_cols = (
                        pd.DataFrame(
                            df["xyz"].apply(DataSet._parse_xyz_counts).tolist()
                        )
                        .fillna(0)
                        .astype(int)
                    )
                    df = pd.concat([df, atom_cols], axis=1)
                    for col in _CALC_COL_RE.findall(select):
                        if col not in df.columns:
                            df[col] = 0

                try:
                    starting_rows = len(df)
                    df = df.query(select)
                    remaining_rows = len(df)
                    if remaining_rows == starting_rows:
                        warning("Selection without effect", select=select)
                    elif remaining_rows == 0:
                        error(
                            "There are no remaining rows",
                            filename=filename,
                            select=select,
                        )
                    else:
                        info(
                            "Applied selection",
                            select=select,
                            remaining_rows=remaining_rows,
                        )
                except Exception as e:
                    error("Failed to apply selection", select=select, error_msg=str(e))

                if mode == "C":
                    df = df.drop(columns=atom_cols.columns)

        if "xyz" not in df.columns:
            error(
                "Required 'xyz' column not found in dataset",
                columns=df.columns.tolist(),
            )

        found_keys = [col for col in df.columns if col != "xyz"]
        info(
            "Dataset columns",
            columns=found_keys,
            total_columns=len(df.columns),
            total_rows=len(df),
        )

        df = df.sample(frac=1).reset_index(drop=True)

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

    @property
    def nuclear_charges(self) -> list[np.ndarray]:
        return [mol.get_atomic_numbers() for mol in self.molecules]

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

    def get_pairwise_features(self, label: str):
        """Return pairwise feature matrix for all molecules.

        Each entry is the sum over all unique atom pairs (i<j) of a
        pairwise function f(Z_i, Z_j, d_ij).

        Args:
            label: Name of pairwise feature. Currently supported: "gCP"
                   (Z_i^2.5 * Z_j^2.5 * exp(-3 * d_ij), inspired by geometric
                   counterpoise correction).

        Returns:
            np.ndarray: Matrix of shape (N, 1) with per-molecule sums.
            list[str]: Feature names.
        """
        _SUPPORTED = ["gCP"]
        if label not in _SUPPORTED:
            error(
                "Unknown pairwise detrending label",
                label=label,
                available=_SUPPORTED,
            )

        features = np.zeros((len(self.molecules), 1))
        for mol_idx, mol in enumerate(self.molecules):
            Z = mol.get_atomic_numbers()
            pos = mol.get_positions()
            n = len(Z)
            i_idx, j_idx = np.triu_indices(n, k=1)
            diffs = pos[i_idx] - pos[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            features[mol_idx, 0] = np.sum(
                Z[i_idx] ** 2.5 * Z[j_idx] ** 2.5 * np.exp(-3 * dists)
            )
        return features, [label]

    @staticmethod
    def _detect_select_mode(select: str | None) -> str:
        """Classify the select expression into one of three loading modes.

        Returns:
            "A": no select expression
            "B": select references only native JSON columns
            "C": select references calculated atom-count columns (n_atoms, n_X)
        """
        if select is None:
            return "A"
        if _CALC_COL_RE.search(select):
            return "C"
        return "B"

    @staticmethod
    def _open_jsonl(filename: str):
        """Return an open file handle for a plain or gzip-compressed JSONL file."""
        with open(filename, "rb") as probe:
            magic = probe.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(filename, "rt")
        return open(filename, "r")

    @staticmethod
    def _reservoir_sample(
        filename: str,
        limit: int,
        select: str | None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """Load up to *limit* rows from a JSONL file using reservoir sampling.

        Processes the file in batches of *batch_size* lines.  Three modes,
        selected automatically from *select*:

        A (no select): reservoir-sample raw line strings; parse JSON only for
                       the final reservoir.
        B (native cols): parse each batch as a DataFrame, apply query, then
                         reservoir-sample surviving rows.
        C (calc cols):  same as B but also compute atom-count columns before
                        querying; drop them before storing in the reservoir.

        Returns a DataFrame with at most *limit* rows.  Atom-count columns are
        never present in the result.
        """
        mode = DataSet._detect_select_mode(select)
        reservoir: list = []
        n_seen = 0

        def _insert(item: object) -> None:
            nonlocal n_seen
            if len(reservoir) < limit:
                reservoir.append(item)
            else:
                j = random.randint(0, n_seen)
                if j < limit:
                    reservoir[j] = item
            n_seen += 1

        def _process_batch(lines: list[str]) -> None:
            if mode == "A":
                for line in lines:
                    _insert(line)
                return

            batch_df = pd.read_json(StringIO("\n".join(lines)), lines=True)

            if mode == "C":
                atom_cols = (
                    pd.DataFrame(
                        batch_df["xyz"].apply(DataSet._parse_xyz_counts).tolist()
                    )
                    .fillna(0)
                    .astype(int)
                )
                batch_df = pd.concat([batch_df, atom_cols], axis=1)
                for col in _CALC_COL_RE.findall(select):
                    if col not in batch_df.columns:
                        batch_df[col] = 0

            batch_df = batch_df.query(select)

            if mode == "C":
                batch_df = batch_df.drop(columns=atom_cols.columns)

            for row in batch_df.to_dict("records"):
                _insert(row)

        with DataSet._open_jsonl(filename) as f:
            batch: list[str] = []
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                batch.append(line)
                if len(batch) >= batch_size:
                    _process_batch(batch)
                    batch = []
            if batch:
                _process_batch(batch)

        if not reservoir:
            return pd.DataFrame()

        if mode == "A":
            return pd.read_json(StringIO("\n".join(reservoir)), lines=True)
        return pd.DataFrame(reservoir)

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
