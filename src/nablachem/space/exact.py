import re
import subprocess
from collections.abc import Iterator
import pathlib

from .utils import AtomStoichiometry, Molecule, Q, SearchSpace, _read_db


class ExactCounter:
    """Python API for exact counts of molecular graphs via surge.

    Note that this implementation avoids the built-in pruning of
    "infeasible" molecular graphs in surge by defining non-standard element labels.

    Uses surge (https://doi.org/10.1186/s13321-022-00604-9) which in turn leverages nauty.

    Requires removal of the following line in surge.c as artificial constraint:
    if (deg[i] + hyd[i] > 4 && hyd[i] > 0) return;
    """

    def __init__(self, binary: str, timeout: int = None):
        """Sets up the environment.

        Parameters
        ----------
        binary : str
            Path to the surge binary.
        timeout : int, optional
            Limits the total runtime for counting any one chemical formula, by default None
        """
        self._binary = binary
        cachedir = pathlib.Path(__file__).parent.resolve() / ".." / "cache"
        self._exact_db = _read_db(cachedir / "space-exact.msgpack.gz")
        self._timeout = timeout

    def _build_cli_arguments(
        self, stoichiometry: AtomStoichiometry, count_only: bool
    ) -> tuple[str, dict[str, str]]:
        max_valence = stoichiometry.largest_valence
        args = [f"-c{max_valence}", f"-d{max_valence}"]
        sf = ""

        letter = "a"
        elements = {}
        used_hydrogen = False

        monovalent_counts = []
        for atom_type, natoms in stoichiometry.components.items():
            if atom_type.valence == 1:
                monovalent_counts.append(natoms)

        for atom_type, natoms in stoichiometry.components.items():
            valence = atom_type.valence
            if (
                valence == 1
                and not used_hydrogen
                and count_only
                and natoms == max(monovalent_counts)
            ):
                elements["H"] = atom_type.label
                sf += f"H{natoms}"
                used_hydrogen = True
            else:
                elements[f"A{letter}"] = atom_type.label
                args.append(f"-EA{letter}{valence}{valence}")
                sf += f"A{letter}{natoms}"
                letter = chr(ord(letter) + 1)

        if count_only:
            extra = "u"
        else:
            extra = "A"

        return " ".join(args) + f" -{extra} " + sf, elements

    def count(self, search_space: SearchSpace, natoms: int) -> int:
        total = 0
        for stoichiometry in search_space.list_cases(natoms):
            total += self.count_one(stoichiometry)
        return total

    def _run(self, args: str, keep_stderr: bool = True) -> str:
        cmd = f"{self._binary} {args}"
        if keep_stderr:
            stderr = subprocess.STDOUT
        else:
            stderr = subprocess.DEVNULL
        stdout = subprocess.check_output(
            cmd, shell=True, stderr=stderr, timeout=self._timeout
        )
        return stdout.decode("utf-8")

    def count_one(self, stoichiometry: AtomStoichiometry):
        # cached?
        label = stoichiometry.canonical_tuple
        if label in self._cache:
            return self._cache[label]

        # run surge
        args, _ = self._build_cli_arguments(stoichiometry, count_only=True)
        stdout = self._run(args)
        try:
            match = [_ for _ in stdout.split("\n") if ">Z generated" in _][0]
        except:
            print(args)
            print(stdout)
            raise NotImplementedError()
        count = int(match.split()[-4])
        self._cache[label] = count
        return count

    def save(self, path):
        with open(path, "w") as f:
            for label, count in self._cache.items():
                f.write(f"{label} {count}\n")

    def load(self, path):
        with open(path) as f:
            for line in f:
                label, count = line.split()
                self._cache[label] = int(count)

    @staticmethod
    def _split_element_label(label: str) -> list[str]:
        """Splits a surge output spec of a stoichiometry into a list of elements.

        Parameters
        ----------
        label : str
            Stoichiometry spec, e.g. "Aa2Ab2Ac". Only works for element labels generated in this class.

        Returns
        -------
        list[str]
            Repeated list of element labels in order, e.g. ["Aa", "Aa", "Ab", "Ab", "Ac"].
        """
        elements = []
        for element, repeat in re.findall("(A[a-z])([0-9]*)", label):
            elements += [element] * int(repeat or 1)
        return elements

    def list_one(self, stoichiometry: AtomStoichiometry) -> Iterator[Molecule]:
        args, lookup = self._build_cli_arguments(stoichiometry, count_only=False)

        stdout = self._run(args, keep_stderr=False)

        # node labels
        node_labels = []
        for component in stoichiometry.components.items():
            node_labels += [component[0].label] * component[1]

        bondtypes = {"-": 1, "=": 2, "#": 3}
        for line in stdout.split("\n"):
            if len(line.strip()) == 0:
                continue

            # parse elements
            elements = [
                lookup[_] for _ in ExactCounter._split_element_label(line.split()[2])
            ]

            # parse edges
            edges = []
            bonds = line.strip().split()[3:]

            for bond in bonds:
                idx1, bondtype, idx2 = re.match(
                    "^([0-9]+)([^0-9])([0-9]+)$", bond
                ).groups()
                order = bondtypes[bondtype]
                edges.append([int(idx1), int(idx2), order])

            yield Molecule(elements, edges)

    def list(
        self, search_space: SearchSpace, natoms: int, selection: Q = None
    ) -> Iterator[Molecule]:
        for stoichiometry in search_space.list_cases(natoms):
            if selection.selected_stoichiometry(stoichiometry):
                yield from self.list_one(stoichiometry)
