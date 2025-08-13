"""Counts how many molecular graphs could be in QM9."""

import nablachem.space as ncs

c = ncs.ApproximateCounter(show_progress=False)
s = ncs.SearchSpace("H:1 C:4 N:3 O:2 F:1")
selection = ncs.Q("C + O + N + F <= 9")

# max number of atoms can be all carbon chain (9 of them) with otherwise H or F
max_natoms = 9 + 2 * 3 + 7 * 2
total = 0
for i in range(1, max_natoms + 1):
    this_i = c.count(s, i, selection)
    print(i, this_i)
    total += this_i

print("Total:", total)
