import nablachem.space as ncs

s = ncs.SearchSpace()
s.add_element(ncs.Element("C", [4]))
s.add_element(ncs.Element("O", [2]))
s.add_element(ncs.Element("N", [3]))
s.add_element(ncs.Element("S", [2, 4, 6]))
s.add_element(ncs.Element("Cl", [1]))
s.add_element(ncs.Element("H", [1]))

c = ncs.ApproximateCounter()
condition = "((((C<=9) & (O <=9)) & (N <=9)) & (S<=9)) & ((H<=28) & (Cl<=9))"

missing = []
natoms = range(18, 25):
pure_only = True
for i in natoms:
    missing += c.missing_parameters(s, i, pure_only, ncs.Q(condition))

for degree_sequence in missing:
    canonical = []
    for degree, natoms in zip(degree_sequence[::2], degree_sequence[1::2]):
        canonical.append(f"{degree}.{natoms}")
    canonical = "_".join(canonical)
    label, ngraphs, average_path_length, success = (
        c.estimate_edit_tree_average_path_length(canonical, 30)
    )
    print(label, average_path_length)
