# enumerates all cases which are prediced to be small by the approximate counter
import nablachem.space as ncs
import hmq


@hmq.task()
def try_count(canonical_label: str):
    c = ncs.ExactCounter("/home/groups/nablachem/opt/surge-linux-v1.0", 5 * 60)
    try:
        return canonical_label, c.count_one(ncs.label_to_stoichiometry(canonical_label))
    except:
        return canonical_label, None


c = ncs.ApproximateCounter()
for estimated in c.estimated_in_cache(maxsize=1e6):
    try_count(estimated)
tag = try_count.submit()
tag.pull(blocking=True)  # wait for results
outputs = {}

for label, count in tag.results():
    if count is None:
        continue
    natoms = ncs.label_to_stoichiometry(label).num_atoms
    if natoms not in outputs:
        outputs[natoms] = open(f"counted-{natoms}.txt", "w")
    outputs[natoms].write(f"{label} {count}\n")
for f in outputs.values():
    f.close()
