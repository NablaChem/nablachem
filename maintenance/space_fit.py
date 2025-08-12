# %%
import numpy as np
import matplotlib.pyplot as plt
import sys

basedir = "../"
sys.path.insert(0, f"{basedir}src")
import nablachem.space as ncs

c = ncs.ApproximateCounter()
db_approx = ncs.ApproximateCounter.read_db(
    f"{basedir}maintenance/space-approx.msgpack.gz"
)
db_compare = ncs.ApproximateCounter.read_db(
    f"{basedir}maintenance/space-compare.msgpack.gz"
)

# styling, fully optional
# brew install --cask font-fira-sans-extra-condensed
# brew install --cask font-fira-sans
from matplotlib import rcParams
from matplotlib import font_manager as fm

for f in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if "Fira" in f:
        fm.fontManager.addfont(f)
fm._load_fontmanager(try_read_cache=False)
rcParams["font.family"] = "Fira Sans Extra Condensed"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.it"] = "Fira Sans Extra Condensed:italic"
# plt.rcParams["mathtext.cal"] = "Typewriter Revo:italic"
rcParams["mathtext.default"] = "regular"
rcParams["mathtext.fontset"] = "custom"
# plt.rcParams['text.latex.preamble'] = r'''
#    \usepackage{amsmath}
#    \usepackage{eulervm}
#    '''

scale = 1.4
rcParams["font.size"] = rcParams["font.size"] * scale

# compute panel D

c = ncs.ApproximateCounter()
#
# pure sequences only
pure_lgs = []
degree_sequences = []
scaling_relation_log_counts = []
for label, count in db_approx.items():
    if ncs._is_pure(label):
        pure_lgs.append(count)
        degrees = sum([[v] * c for v, c in zip(label[::2], label[1::2])], [])
        degree_sequences.append(degrees)
        scaling_relation_log_counts.append(
            float(
                c._count_one_asymptotically_log(
                    degrees=tuple(degrees), calibrated=False
                )
            )
        )

# %%
f, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=600, constrained_layout=True)

# despine
for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
# letters
for i, label in zip(axs, ["A", "B", "C", "D"]):
    i.set_title(label, loc="left", fontweight="bold")


####### A
counts = []
lgs = []
cases = []
for k, (count, lg) in db_compare.items():
    counts.append(count)
    lgs.append(lg)
    cases.append(k)
counts = np.array(counts)
lgs = np.array(lgs)
axs[0].hexbin(
    lgs, np.log(counts), gridsize=(40, 30), lw=0, extent=(0, 30, 0, 25), mincnt=1
)

powers = np.arange(0, 10, 3)
ypos = np.log(10**powers)
y_labels = [f"$\mathdefault{{10^{{{int(tick)}}}}}$" for tick in powers]
axs[0].set_yticks(ypos)
axs[0].set_yticklabels(y_labels)
axs[0].set_ylabel("number of graphs $\mathdefault{\it{n}}$")
axs[0].set_xlabel("average path length $\mathdefault{\it{l_G}}$")
axs[0].set_ylim(0, 24)


def score(params):
    poly = np.poly1d(params)
    counts_predicted = np.exp(poly(lgs))
    counts_predicted[counts_predicted < 0] = 1
    ratios = np.log(counts_predicted / counts)
    absratios = np.abs(ratios)
    return len(absratios[absratios > np.log(10)]) / len(ratios)


from scipy.optimize import minimize

p0 = (1.0, -1)  # initial guess
result = minimize(score, p0, method="Nelder-Mead")
poly = np.poly1d(result.x)
print("fitting results panel A (slope, offset)", result.x)
print("total protomolecule count panel A", sum(counts))
xs = np.linspace(0, 25, 100)
axs[0].plot(xs, poly(xs), color="red", label=f"linear fit")
axs[0].legend(loc="lower right", frameon=False)

###### B
coeffs = result.x
poly = np.poly1d(coeffs)
counts_predicted = np.exp(poly(lgs))
counts_predicted[counts_predicted < 0] = 1
data = np.abs(np.log(counts / counts_predicted))
axs[1].hist(
    data,
    density=True,
    cumulative=True,
    bins=100,
    range=(0, 15),
    histtype="step",
)
axs[1].set_xlim(0, np.log(200))
xticks = (2, 10, 100)


def detailline(x, y):
    axs[1].plot((0, x, x), (y, y, 0), color="black", lw=0.5, alpha=0.5)


for xtick in xticks:
    share = len(data[data < np.log(xtick)]) / len(data)
    axs[1].text(0, share, f"{share:0.2f}", ha="left")
    detailline(np.log(xtick), share)

axs[1].set_xticks(np.log(xticks), xticks)
axs[1].set_xlabel("relative size error per stoichiometry")
axs[1].set_ylabel("cumulative distribution")

###### C

expected_pure_lgs = []
actual_nonpure_lgs = []
for k, v in db_approx.items():
    if not ncs._is_pure(k):
        pure_k = ncs._to_pure(k)
        try:
            expected_pure_lgs.append(db_approx[pure_k] * c._pure_prefactor(k))
            actual_nonpure_lgs.append(v)
        except KeyError:
            # dont have the pure version
            continue
    else:
        scaling = c._pure_prefactor(k)
        assert scaling == 1


axs[2].hexbin(
    actual_nonpure_lgs,
    expected_pure_lgs,
    gridsize=(50, 35),
    lw=0,
    mincnt=1,
)
axs[2].set_xlabel("actual path length $\mathdefault{\it{l_G}}$")
axs[2].set_ylabel("approximate path length $\mathdefault{\it{l_G}}$")
axs[2].plot([5, 60], [5, 60], color="red", label="ideal")
axs[2].legend(loc="lower right", frameon=False)
axs[2].set_xlim(5, 65)
axs[2].set_ylim(5, 65)
print("number of data points in panel A, B", len(lgs))
print("number of data points in panel C", len(actual_nonpure_lgs))

###### D


pure_lgs = np.array(pure_lgs)
scaling_relation_log_counts = np.array(scaling_relation_log_counts)
atom_counts = np.array([len(_) for _ in degree_sequences])
M = np.array([sum(_) for _ in degree_sequences])
kmax = np.array([max(_) for _ in degree_sequences])
threshold = 20

mask = atom_counts > threshold
axs[3].hexbin(
    scaling_relation_log_counts[mask], pure_lgs[mask], gridsize=(40, 20), lw=0, mincnt=1
)

axs[3].plot((0, 100), (0, 100), color="red", label="ideal")
axs[3].set_xlabel("asymptotic scaling relation log($\mathdefault{\it{G}}$)")
axs[3].set_ylabel("average path length $\mathdefault{\it{l_G}}$")
coeffs = np.polyfit(scaling_relation_log_counts[mask], pure_lgs[mask], 1)
poly = np.poly1d(coeffs)
print(
    f"fitted on all {len(scaling_relation_log_counts[mask])} points with more than {threshold} atoms",
)
print("coefficients fit panel D", coeffs)
axs[3].plot(
    scaling_relation_log_counts[mask],
    poly(scaling_relation_log_counts[mask]),
    color="orange",
    label="linear fit",
)
axs[3].set_xlim(0, 180)
axs[3].set_ylim(0, 110)
axs[3].legend(frameon=False)

# %%
plt.savefig("space_fit.pdf", bbox_inches="tight")
