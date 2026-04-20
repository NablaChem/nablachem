#!/usr/bin/env python3
# %% imports
import re
import sys
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
# show_median     : plot the median line
# show_individual : plot per-run/per-file light traces
# run_idx         : None = all runs; int = only that run (ignored in aggregate mode)
# aggregate_files : True = load _1.npz–_5.npz, compute per-file median curves,
#                   interpolate onto a common x-grid, then take the median across files

show_median = True
show_individual = True
run_idx = True
aggregate_files = True

# %% load & derive

path = next(
    (a for a in sys.argv[1:] if a.endswith(".npz")),
    "/Users/ali/second_project_data/results/out_cMBDFLocal_elemental_2048_1.npz",
)


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


if aggregate_files:
    _stem = re.sub(r"_\d+\.npz$", "", path)
    _ds = [np.load(f"{_stem}_{i}.npz") for i in range(1, 6)]
    print(f"Loaded {len(_ds)} files")

    # Per-file: compute median improvement curve, then dedup
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

    # Interpolate onto common x-grid (np.interp requires increasing x → flip)
    _all_x = np.sort(np.unique(np.concatenate([c[0] for c in _file_curves])))
    _mat = np.array([np.interp(_all_x, xc[::-1], yc[::-1]) for xc, yc in _file_curves])
    n_dims_at_eval = _all_x[::-1]
    impr_median_test = np.median(_mat, axis=0)[::-1]
    q25_test = np.percentile(_mat, 25, axis=0)[::-1]
    q75_test = np.percentile(_mat, 75, axis=0)[::-1]
    n_runs = len(_file_curves)

    # SI panels: use first file as representative
    _d0 = _ds[0]
    runs_to_plot = list(range(_d0["test_errors"].shape[0]))
    weight_log_si = _d0["weight_log"]
    value_log_si = _d0["value_log"]
    median_weights = np.median(weight_log_si, axis=0)
    median_value = np.median(value_log_si, axis=0)
    running_avg = np.cumsum(median_value) / (np.arange(len(median_value)) + 1)
    n_dims = (median_weights > 0.001).sum(axis=1)
    label_suffix = f"{len(_ds)} files (aggregated)"

else:
    _d = np.load(path)
    steps = _d["rmse_steps"]  # (n_eval_steps,)
    test_errors = _d["test_errors"]  # (n_runs, n_eval_steps)
    val_errors = _d["val_errors"]  # (n_runs, n_eval_steps)
    weight_log_si = _d["weight_log"]  # (n_runs, total_steps, n_features)
    value_log_si = _d["value_log"]  # (n_runs, total_steps)
    n_runs = test_errors.shape[0]

    if run_idx is not None and run_idx >= n_runs:
        print(
            f"Warning: run_idx={run_idx} out of range for {n_runs} run(s); using all runs."
        )
        run_idx = None

    runs_to_plot = [run_idx] if run_idx is not None else list(range(n_runs))
    median_test = np.median(test_errors[runs_to_plot], axis=0)
    median_weights = np.median(weight_log_si[runs_to_plot], axis=0)
    median_value = np.median(value_log_si[runs_to_plot], axis=0)

    ref_test = median_test[0]
    impr_median_test = (ref_test - median_test) / ref_test * 100
    impr_all_test = (ref_test - test_errors[runs_to_plot]) / ref_test * 100

    median_val = np.median(val_errors[runs_to_plot], axis=0)
    ref_val = median_val[0]
    impr_median_val = (ref_val - median_val) / ref_val * 100

    q25_test = np.percentile(impr_all_test, 25, axis=0)
    q75_test = np.percentile(impr_all_test, 75, axis=0)

    running_avg = np.cumsum(median_value) / (np.arange(len(median_value)) + 1)
    n_dims = (median_weights > 0.001).sum(axis=1)
    n_dims_runs = (weight_log_si[runs_to_plot] > 0.001).sum(axis=2)

    n_dims_at_eval = n_dims[steps]
    n_dims_runs_at_eval = n_dims_runs[:, steps]

    # Deduplicate: select point with highest validation improvement per n_dims
    _seen = {}
    for i in range(1, len(n_dims_at_eval)):
        d_val = int(n_dims_at_eval[i])
        if d_val not in _seen or impr_median_val[i] > impr_median_val[_seen[d_val]]:
            _seen[d_val] = i
    _keep = np.array(sorted({0} | set(_seen.values())))

    n_dims_at_eval = n_dims_at_eval[_keep]
    impr_median_test = impr_median_test[_keep]
    impr_all_test = impr_all_test[:, _keep]
    q25_test = q25_test[_keep]
    q75_test = q75_test[_keep]
    n_dims_runs_at_eval = n_dims_runs_at_eval[:, _keep]

    label_suffix = (
        f"run {run_idx}"
        if run_idx is not None
        else f"{n_runs} run{'s' if n_runs > 1 else ''}"
    )

# %% main figure (paper)

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

# -- per-run / per-file light traces
if show_individual and n_runs > 1 and not aggregate_files:
    for i in range(len(runs_to_plot)):
        ax_main.plot(
            n_dims_runs_at_eval[i],
            impr_all_test[i],
            color="C0",
            alpha=0.15,
            linewidth=0.7,
        )

# -- IQR band
if show_median and n_runs > 1 and not aggregate_files:
    ax_main.fill_between(
        n_dims_at_eval, q25_test, q75_test, color="C0", alpha=0.15, label="IQR"
    )

# -- median line
if show_median:
    ax_main.plot(
        n_dims_at_eval,
        impr_median_test,
        marker="o",
        markersize=4,
        color="C0",
        linewidth=1.5,
        label="Test RMSE (median)",
    )

ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax_main.axhspan(-10, 0, color="gray", alpha=0.15, zorder=0)
ax_main.set_ylim(20, -10)
ax_main.set_xlim(0, n_dims_at_eval[0])
ax_main.set_xlabel("Number of features")
ax_main.set_ylabel("Relative improvement (RMSE %)")
ax_main.legend(loc="lower left")

plt.tight_layout()
plt.savefig(path.replace(".npz", "_main.pdf"), bbox_inches="tight")
print(f"Saved {path.replace('.npz', '_main.pdf')}")
plt.show()
# %% SI figure (3 panels)

fig_si, (ax_w, ax_v, ax_d) = plt.subplots(1, 3, figsize=(13.5, 4.5))
fig_si.suptitle(
    f"Supplementary — {path}  ({label_suffix})",
    fontweight="bold",
)

for _ax, _letter in zip((ax_w, ax_v, ax_d), ("a)", "b)", "c)")):
    format_panel(_ax)
    _ax.annotate(
        _letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=_label_fs,
        fontweight="bold",
    )

# -- weights
if show_individual:
    for r in runs_to_plot:
        ax_w.plot(
            weight_log_si[r],
            alpha=0.15 if show_median else 0.4,
            linewidth=0.4,
            color="steelblue",
        )
if show_median:
    ax_w.plot(median_weights, alpha=0.8, linewidth=0.9, color="steelblue")
ax_w.set_title(f"Feature Weights ({'median' if show_median else 'per-run'})")
ax_w.set_ylabel("Weight")
ax_w.set_xlabel("Step")
ax_w.grid(True, alpha=0.3)

# -- LQ value log
if show_individual:
    for r in runs_to_plot:
        ax_v.semilogy(
            value_log_si[r],
            color=f"C{r % 10}",
            alpha=0.3 if show_median else 0.7,
            linewidth=0.6,
            label=f"Run {r}" if not show_median else None,
        )
if show_median:
    ax_v.semilogy(median_value, color="C3", alpha=0.5, linewidth=0.8, label="median")
    ax_v.semilogy(running_avg, color="C1", linewidth=1.5, label="running avg")
ax_v.set_title("Low-Quality Val Error")
ax_v.set_ylabel("RMSE (log)")
ax_v.set_xlabel("Step")
ax_v.legend(fontsize=8)
ax_v.grid(True, alpha=0.3)

# -- active dims
if show_individual:
    for r in runs_to_plot:
        n_dims_r = (weight_log_si[r] > 0.001).sum(axis=1)
        ax_d.plot(
            n_dims_r,
            color=f"C{r % 10}",
            alpha=0.3 if show_median else 0.8,
            linewidth=0.8,
            label=f"Run {r}" if not show_median else None,
        )
if show_median:
    ax_d.plot(n_dims, color="C2", linewidth=1.5, label="median")
ax_d.set_title("Active Dimensions")
ax_d.set_ylabel("Count")
ax_d.set_xlabel("Step")
ax_d.grid(True, alpha=0.3)
if not show_median and show_individual and len(runs_to_plot) > 1:
    ax_d.legend(fontsize=8)

plt.tight_layout()
plt.savefig(path.replace(".npz", "_SI.pdf"), bbox_inches="tight")
print(f"Saved {path.replace('.npz', '_SI.pdf')}")
plt.show()

# %%
