# %%
import msgpack
import glob
import gzip


def label_to_dbkey(label: str):
    return tuple([int(_) for _ in label.replace("_", ".").split(".")])


def db_to_file(db, label):
    print(f"storing {len(db)} entries to {label}")
    with gzip.open(f"space-{label}.msgpack.gz", "wb") as f:
        msgpack.dump(db, f, use_bin_type=True)


def getlines(fn):
    if fn.endswith(".gz"):
        with gzip.open(fn) as fh:
            lines = [_.decode("ascii") for _ in fh.readlines()]
    else:
        with open(fn) as fh:
            lines = fh.readlines()
    return lines


db_exact = {}
db_approx = {}

for fn in glob.glob("../database/space-*"):
    lines = getlines(fn)
    is_pathlength = "." in lines[0].split()[1]

    for line in lines:
        parts = line.split()
        label = label_to_dbkey(parts[0])
        if is_pathlength:
            db_approx[label] = float(parts[1])
        else:
            db_exact[label] = int(parts[1])

db_export_approx = {}
db_compare = {}
for k, v in db_approx.items():
    if k in db_exact:
        db_compare[k] = (db_exact[k], v)
    else:
        db_export_approx[k] = v

db_to_file(db_exact, "exact")
db_to_file(db_export_approx, "approx")
db_to_file(db_compare, "compare")
