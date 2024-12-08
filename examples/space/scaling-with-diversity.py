import nablachem.space as ncs

c = ncs.ApproximateCounter(show_progress=False)
s = ncs.SearchSpace()
labels = ""
elements = [
    ncs.Element("C", [4]),
    ncs.Element("H", [1]),
    ncs.Element("O", [2]),
    ncs.Element("N", [3, 5]),
    ncs.Element("F", [1]),
    ncs.Element("S", [2, 4, 6]),
    ncs.Element("P", [3, 5]),
    ncs.Element("Cl", [1]),
    ncs.Element("Br", [1]),
    ncs.Element("I", [1]),
    ncs.Element("Si", [4]),
]
natoms = 7
print("# atoms, elements, # sum formulas, # molecules")
for element in elements:
    labels += element.label
    s.add_element(element)
    print(
        natoms,
        labels,
        c.count_sum_formulas(s, natoms),
        c.count(s, natoms),
    )
