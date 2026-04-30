#!/usr/bin/env python3
# %% imports
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm


# %% plot style (from NablaChem/nablachem maintenance/plotstyle.py)
def load_style(scale: float = 1.4):
    fm._load_fontmanager(try_read_cache=False)
    for f in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        if "Fira" in f:
            fm.fontManager.addfont(f)

    _preferred = "Fira Sans Extra Condensed"
    _fallback = "Fira Sans"
    _available = fm.get_font_names()
    _font = (
        _preferred
        if _preferred in _available
        else (_fallback if _fallback in _available else "DejaVu Sans")
    )
    print(f"[load_style] using font: {_font}")

    rcParams["font.family"] = _font
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.it"] = f"{_font}:italic"
    rcParams["mathtext.default"] = "regular"

    base = 10.0
    rcParams["font.size"] = base * scale
    rcParams["axes.labelsize"] = base * scale
    rcParams["xtick.labelsize"] = base * scale
    rcParams["ytick.labelsize"] = base * scale
    rcParams["legend.fontsize"] = base * scale * 0.9
    rcParams["legend.frameon"] = False
    rcParams["legend.handletextpad"] = 0.1
    rcParams["legend.handlelength"] = 1.0

    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.size"] = 4
    rcParams["ytick.major.size"] = 4
    rcParams["xtick.minor.size"] = 2
    rcParams["ytick.minor.size"] = 2


def format_panel(ax, letter: str = ""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    if letter:
        ax.set_title(letter, loc="left", fontweight="bold")


load_style()

_label_fs = rcParams["axes.labelsize"] * 1.2

# %% config
show_median = True
show_individual = True
run_idx = True
aggregate_files = True

# %% representations to compare — edit paths as needed
REPRESENTATIONS = [
    (
        "/Users/ali/second_project_data/results/out_FCHL19Local_elemental_2048_1.npz",
        "FCHL19",
        "C0",
    ),
    (
        "/Users/ali/second_project_data/results/out_MACELocal_local_2048_1.npz",
        "MACE",
        "C1",
    ),
    (
        "/Users/ali/second_project_data/results/out_cMBDFLocal_elemental_2048_1.npz",
        "cMBDF",
        "C2",
    ),
]


# %% loading helpers


def _dedup(nd, impr, select_by=None):
    """Keep the highest-val-improvement point per unique n_dims; always keep index 0 (reference)."""
    if select_by is None:
        select_by = impr
    seen = {}
    for i in range(1, len(nd)):
        v = int(nd[i])
        if v not in seen or select_by[i] > select_by[seen[v]]:
            seen[v] = i
    keep = np.array(sorted({0} | set(seen.values())))
    return nd[keep], impr[keep]


def load_repr(path):
    """Load and aggregate one representation; return curve arrays for the main plot."""
    if aggregate_files:
        _stem = re.sub(r"_\d+\.npz$", "", path)
        _ds = [np.load(f"{_stem}_{i}.npz") for i in range(1, 6)]
        print(f"[{path}] Loaded {len(_ds)} files")

        _file_curves = []
        for _d in _ds:
            _s = _d["rmse_steps"]
            _mte = np.median(_d["test_errors"], axis=0)
            _impr = (_mte[0] - _mte) / _mte[0] * 100
            _mve = np.median(_d["val_errors"], axis=0)
            _impr_val = (_mve[0] - _mve) / _mve[0] * 100
            _mw = np.median(_d["weight_log"], axis=0)
            _nd = (_mw > 0.001).sum(axis=1)[_s]
            _file_curves.append(_dedup(_nd, _impr, select_by=_impr_val))

        _all_x = np.sort(np.unique(np.concatenate([c[0] for c in _file_curves])))
        _mat = np.array(
            [np.interp(_all_x, xc[::-1], yc[::-1]) for xc, yc in _file_curves]
        )
        n_dims_at_eval = _all_x[::-1]
        impr_median_test = np.median(_mat, axis=0)[::-1]
        q25_test = np.percentile(_mat, 25, axis=0)[::-1]
        q75_test = np.percentile(_mat, 75, axis=0)[::-1]
        n_runs = len(_file_curves)
    else:
        _d = np.load(path)
        steps = _d["rmse_steps"]
        test_errors = _d["test_errors"]
        val_errors = _d["val_errors"]
        weight_log = _d["weight_log"]
        n_runs = test_errors.shape[0]

        _ri = run_idx if (run_idx is not None and run_idx < n_runs) else None
        _runs = [_ri] if _ri is not None else list(range(n_runs))

        median_test = np.median(test_errors[_runs], axis=0)
        median_weights = np.median(weight_log[_runs], axis=0)
        median_val = np.median(val_errors[_runs], axis=0)

        ref_test = median_test[0]
        impr_median_test = (ref_test - median_test) / ref_test * 100
        impr_all_test = (ref_test - test_errors[_runs]) / ref_test * 100

        ref_val = median_val[0]
        impr_median_val = (ref_val - median_val) / ref_val * 100

        q25_test = np.percentile(impr_all_test, 25, axis=0)
        q75_test = np.percentile(impr_all_test, 75, axis=0)

        n_dims = (median_weights > 0.001).sum(axis=1)
        n_dims_at_eval = n_dims[steps]

        _seen = {}
        for i in range(1, len(n_dims_at_eval)):
            d_val = int(n_dims_at_eval[i])
            if d_val not in _seen or impr_median_val[i] > impr_median_val[_seen[d_val]]:
                _seen[d_val] = i
        _keep = np.array(sorted({0} | set(_seen.values())))

        n_dims_at_eval = n_dims_at_eval[_keep]
        impr_median_test = impr_median_test[_keep]
        q25_test = q25_test[_keep]
        q75_test = q75_test[_keep]

    return n_dims_at_eval, impr_median_test, q25_test, q75_test, n_runs


# %% main figure — all representations on one plot

fig_main, ax_main = plt.subplots(1, 1, figsize=(10, 7))
format_panel(ax_main)
ax_main.annotate(
    "a)",
    xy=(0.0, 1.0),
    xycoords="axes fraction",
    xytext=(0, 6),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=_label_fs,
    fontweight="bold",
)

_max_x = 0
for _path, _label, _color in REPRESENTATIONS:
    try:
        nd, impr, q25, q75, n_runs = load_repr(_path)
    except FileNotFoundError:
        print(f"[skip] {_path} not found")
        continue

    # ax_main.fill_between(nd, q25, q75, color=_color, alpha=0.15)
    ax_main.plot(
        nd,
        impr,
        marker="o",
        markersize=4,
        color=_color,
        linewidth=1.5,
        label=_label,
    )
    _max_x = max(_max_x, nd[0])

ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax_main.axhspan(-10, 0, color="gray", alpha=0.15, zorder=0)
ax_main.set_ylim(30, -10)
ax_main.set_xlim(0, _max_x)
ax_main.set_xlabel("Number of features")
ax_main.set_ylabel("Relative improvement (RMSE %)")
ax_main.legend(loc="lower left")

plt.tight_layout()
_out = "/Users/ali/second_project_data/results/comparison_main.pdf"
plt.savefig(_out, bbox_inches="tight")
print(f"Saved {_out}")
plt.show()

# %%
