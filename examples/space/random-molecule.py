import nablachem.space as ncs

counter = ncs.ApproximateCounter(show_progress=False)
space = ncs.SearchSpace("C:4 H:1 O:1 N:3 F:1")

mols = ncs.random_sample(
    counter, space, natoms=10, nmols=3, selection=ncs.Q("C + O + N + F <= 9")
)

for mol in mols:
    print(mol.SMILES)
