import nablachem.space as ncs

c = ncs.ApproximateCounter(show_progress=False)
s = ncs.SearchSpace()
s.add_element(ncs.Element("C", [4]))
s.add_element(ncs.Element("H", [1]))
s.add_element(ncs.Element("O", [2]))
s.add_element(ncs.Element("N", [3]))
s.add_element(ncs.Element("F", [1]))

c.random_sample(s, natoms=10, nmols=3, selection=ncs.Q("C + O + N + F <= 9"))
c.count(s, 6, ncs.Q("C + O + N + F <= 9"))
c.count(s, 30, ncs.Q("(C > 10) & (H > 10) & (H + F > 15)"))
