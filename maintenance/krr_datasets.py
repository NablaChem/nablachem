# local copy of all result files are on the uni kassel cluster in /home/groups/nablachem/DATA/krr_datasets
import json
import gzip


def JSONL_from_MultixcQM9(outfile):
    """Extract MultixcQM9 dataset and save as gzipped JSONL file with PBE/TZP and GFNXTB energies in hartree"""
    import numpy as np
    from openqdc.datasets import MultixcQM9
    from ase.units import kcal, mol

    print("Loading MultixcQM9 dataset...")
    dataset = MultixcQM9(
        energy_unit="hartree",
        distance_unit="ang",
        array_format="numpy",
    )

    energy_names = dataset.energy_target_names
    rows = []

    print(f"Processing {len(dataset)} molecules...")
    for i in range(len(dataset)):
        molecule = dataset[i]

        # Extract atomic numbers and positions
        atomic_numbers = molecule["atomic_numbers"].ravel().astype(int)
        positions = molecule["positions"].astype(float)

        # Get energies and create mapping
        energies = molecule["energies"].ravel().astype(float)
        energy_mapping = dict(zip(energy_names, energies))

        # Construct XYZ string
        natoms = len(atomic_numbers)
        xyz_lines = [str(natoms), ""]  # natoms and empty comment line

        # Map atomic numbers to symbols
        atom_symbols = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

        for j in range(natoms):
            z = atomic_numbers[j]
            symbol = atom_symbols.get(z, f"Z{z}")  # fallback for unknown elements
            x, y, z_coord = positions[j]
            xyz_lines.append(f"{symbol} {x:.6f} {y:.6f} {z_coord:.6f}")

        xyz_string = "\n".join(xyz_lines)

        # Create row with both energy targets
        row = {"xyz": xyz_string} | energy_mapping
        rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} molecules...")

    # Write to JSONL gzipped file
    print(f"Writing {len(rows)} molecules to {outfile}...")
    with gzip.open(outfile, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Successfully wrote MultixcQM9 dataset to {outfile}")
    return len(rows)


def JSONL_from_qm9(outfile):
    import numpy as np
    import requests
    import io
    import tarfile
    import os

    """Fetch QM9 dataset and extract molecular geometries and energies."""

    # Check if uncharacterized.txt exists locally
    if os.path.exists("uncharacterized.txt"):
        print("Found local uncharacterized.txt, using it...")
        with open("uncharacterized.txt", "rb") as f:
            content = f.read()
    else:
        print("Downloading uncharacterized.txt...")
        res = requests.get("https://ndownloader.figshare.com/files/3195404")
        content = res.content

    exclusion_ids = [
        _.strip().split()[0] for _ in content.decode("ascii").split("\n")[9:-2]
    ]

    # Check if dsgdb9nsd.xyz.tar.bz2 exists locally
    if os.path.exists("dsgdb9nsd.xyz.tar.bz2"):
        print("Found local dsgdb9nsd.xyz.tar.bz2, using it...")
        with open("dsgdb9nsd.xyz.tar.bz2", "rb") as f:
            tar_content = f.read()
    else:
        print("Downloading dsgdb9nsd.xyz.tar.bz2...")
        res = requests.get("https://ndownloader.figshare.com/files/3195389")
        tar_content = res.content

    webfh = io.BytesIO(tar_content)
    t = tarfile.open(fileobj=webfh)

    rows = []
    for fo in t:
        lines = t.extractfile(fo).read().decode("ascii").split("\n")
        natoms = int(lines[0])

        lines = lines[: 2 + natoms]
        molid = lines[1].strip().split()[0]
        if molid in exclusion_ids:
            continue
        columns = "A B C mu alpha homo lumo _ r2 zpve _ U H G Cv"
        parts = lines[1].strip().split()

        def fixfloat(s):
            return float(s.replace("*^", "e"))

        energy = fixfloat(parts[12])
        others = {}
        for idx, column in enumerate(columns.split()):
            if column == "_":
                continue
            others[column] = fixfloat(parts[idx + 2])

        # Construct proper XYZ format
        natoms = int(lines[0])
        xyz_lines = [str(natoms), ""]  # natoms and empty comment line
        xyz_lines.extend(lines[2 : 2 + natoms])  # atom lines
        xyz_string = "\n".join(xyz_lines).replace("*^", "e")

        atomref = {
            "H": -0.500273,
            "C": -37.846772,
            "N": -54.583861,
            "O": -75.064579,
            "F": -99.718730,
        }
        rows.append(
            {
                "xyz": xyz_string,
                "totalenergy": energy,
                "atomisation_energy": energy
                - sum(xyz_string.count(el) * ref for el, ref in atomref.items()),
            }
            | others
        )

    # Write to JSONL gzipped file
    print(f"Writing {len(rows)} molecules to {outfile}...")
    with gzip.open(outfile, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Successfully wrote QM9 dataset to {outfile}")
    return len(rows)


def JSONL_from_GDB_BSIE(outfile):
    """Extract GDB-BSIE dataset from local tar file and save as gzipped JSONL file"""
    import tarfile
    import pandas as pd
    import numpy as np
    import json
    import gzip
    import os

    if not os.path.exists("GDB-BSIE.tar"):
        raise FileNotFoundError("GDB-BSIE.tar not found in current directory")

    print("Loading GDB-BSIE dataset from GDB-BSIE.tar...")

    with tarfile.open("GDB-BSIE.tar", "r") as tar:
        # Read energies dataframe
        def _read_txt(prefix, fn):
            member = tar.getmember(fn)
            f = tar.extractfile(member)
            return pd.read_csv(
                f,
                delim_whitespace=True,
                index_col=None,
                skiprows=1,
                names=f"index {prefix}sto-3g {prefix}cc-pvdz {prefix}cc-pvtz {prefix}cc-pvqz {prefix}cc-pv5z".split(),
            )

        energies_hf_df = _read_txt("rhf", "Energies/RHF/energies.txt")
        energies_b3lyp_df = _read_txt("b3lyp", "Energies/B3LYP/energies.txt")

        # Parse all XYZ files
        element_list = ["H", "C", "N", "O", "F", "S", "Cl"]
        xyz_data = []
        for member in tar.getmembers():
            if member.name.startswith("XYZ/") and member.name.endswith(".xyz"):
                # Extract molecule index from filename
                filename = member.name.split("/")[-1]
                mol_index = int(filename.replace(".xyz", ""))

                # Read XYZ content
                xyz_file = tar.extractfile(member)
                xyz_content = xyz_file.read().decode("utf-8")
                lines = xyz_content.strip().split("\n")

                # Parse header
                natoms = int(lines[0])
                commentdata = json.loads(lines[1])

                # Extract geometry lines
                geometry_lines = lines[2 : 2 + natoms]

                # Create proper XYZ string format
                xyz_string = f"{natoms}\n\n" + "\n".join(geometry_lines)

                # Count elements
                elements = [line.strip().split()[0] for line in geometry_lines]
                counts = {f"element_{el}": elements.count(el) for el in element_list}
                if sum(counts.values()) != len(elements):
                    raise ValueError("Element count mismatch.")

                row = {
                    "index": mol_index,
                    "xyz_geometry": "\n".join(geometry_lines),
                    "natoms": len(elements),
                    **counts,
                    **commentdata,
                }
                xyz_data.append(row)

        # Create XYZ dataframe
        xyz_df = pd.DataFrame(xyz_data)

        # Merge dataframes on index
        merged_df = energies_hf_df.merge(xyz_df, on="index", how="left")
        merged_df = merged_df.merge(energies_b3lyp_df, on="index", how="left")

    print(f"Processing {len(merged_df)} molecules...")

    rows = []
    for idx, row in merged_df.iterrows():
        # Construct proper XYZ string
        natoms = row["natoms"]
        xyz_lines = [str(natoms), ""]  # natoms and empty comment line
        xyz_lines.extend(row["xyz_geometry"].split("\n"))
        xyz_string = "\n".join(xyz_lines)

        # Create row with energies
        json_row = {
            "xyz": xyz_string,
            "RHF/STO-3G": float(row["rhfsto-3g"]),
            "RHF/cc-pVDZ": float(row["rhfcc-pvdz"]),
            "RHF/cc-pVTZ": float(row["rhfcc-pvtz"]),
            "RHF/cc-pVQZ": float(row["rhfcc-pvqz"]),
            "RHF/cc-pV5Z": float(row["rhfcc-pv5z"]),
            "B3LYP/STO-3G": float(row["b3lypsto-3g"]),
            "B3LYP/cc-pVDZ": float(row["b3lypcc-pvdz"]),
            "B3LYP/cc-pVTZ": float(row["b3lypcc-pvtz"]),
            "B3LYP/cc-pVQZ": float(row["b3lypcc-pvqz"]),
            "B3LYP/cc-pV5Z": float(row["b3lypcc-pv5z"]),
            "geoindex": int(row["geometry_index"]),
            "index": int(row["index"]),
        }

        rows.append(json_row)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} molecules...")

    # Write to JSONL gzipped file
    print(f"Writing {len(rows)} molecules to {outfile}...")
    with gzip.open(outfile, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Successfully wrote GDB-BSIE dataset to {outfile}")
    return len(rows)


JSONL_from_GDB_BSIE("gdb_bsie.jsonl.gz")
JSONL_from_qm9("qm9.jsonl.gz")
JSONL_from_MultixcQM9("multixcqm9.jsonl.gz")
