#!/usr/bin/env python3
# %% imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# %% config
# show_median   : plot the median across runs
# show_individual : plot per-run lines (all runs, or only `run_idx` if set)
# run_idx       : None = all runs; int = only that run index (0-based)

show_median = True
show_individual = True
run_idx = 2  # e.g. 0 to plot only the first run

# %% load

path = next(
    (a for a in sys.argv[1:] if a.endswith(".npz")),
    "/Users/ali/second_project_data/results/round_one_results/out_cMBDFLocal_elemental_512.npz",
)
d = np.load(path)

steps = d["rmse_steps"]  # (n_eval_steps,)
test_errors = d["test_errors"]  # (n_runs, n_eval_steps)
val_errors = d["val_errors"]  # (n_runs, n_eval_steps)
value_log = d["value_log"]  # (n_runs, n_steps)
weight_log = d["weight_log"]  # (n_runs, n_steps, 40)

n_runs = test_errors.shape[0]

# %% derived

runs_to_plot = [run_idx] if run_idx is not None else list(range(n_runs))

median_test = np.median(test_errors[runs_to_plot], axis=0)
median_val = np.median(val_errors[runs_to_plot], axis=0)
median_value = np.median(value_log[runs_to_plot], axis=0)
median_weights = np.median(weight_log[runs_to_plot], axis=0)

ref_test = median_test[0]
ref_val = median_val[0]
impr_median_test = (ref_test - median_test) / ref_test * 100
impr_median_val = (ref_val - median_val) / ref_val * 100

# per-run improvement arrays
impr_all_test = (
    (ref_test - test_errors[runs_to_plot]) / ref_test * 100
)  # (n_runs, n_steps)

# IQR across runs
q25_test = np.percentile(impr_all_test, 25, axis=0)
q75_test = np.percentile(impr_all_test, 75, axis=0)

# running best (cumulative max of median improvement)
running_best_test = np.maximum.accumulate(impr_median_test)

running_avg = np.cumsum(median_value) / (np.arange(len(median_value)) + 1)
n_dims = (median_weights > 0.001).sum(axis=1)

# %% figure

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.3)
ax_main = fig.add_subplot(gs[:, 0])
ax_w = fig.add_subplot(gs[0, 1])
ax_v = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[2, 1])

label_suffix = (
    f"run {run_idx}"
    if run_idx is not None
    else f"{n_runs} run{'s' if n_runs > 1 else ''}"
)
fig.suptitle(
    f"Cluster results — {path}  ({label_suffix})",
    fontsize=14,
    fontweight="bold",
)

# -- per-run lines (light traces behind median)
if show_individual and n_runs > 1:
    for i, r in enumerate(runs_to_plot):
        ax_main.plot(
            steps,
            impr_all_test[i],
            color="C0",
            alpha=0.15,
            linewidth=0.7,
        )

# -- IQR band
if show_median and n_runs > 1:
    ax_main.fill_between(
        steps, q25_test, q75_test, color="C0", alpha=0.15, label="IQR (runs)"
    )

# -- median line
if show_median:
    ax_main.plot(
        steps,
        impr_median_test,
        marker="o",
        markersize=4,
        color="C0",
        linewidth=1.5,
        label="Test RMSE (median)",
    )
    ax_main.plot(
        steps,
        impr_median_val,
        marker="s",
        markersize=3,
        color="C1",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label="Val RMSE (median)",
    )
    # running best envelope
    ax_main.plot(
        steps,
        running_best_test,
        color="C2",
        linewidth=1.5,
        linestyle=":",
        label="Best so far (test)",
    )
    # annotate the best step
    best_idx = np.argmax(impr_median_test)
    ax_main.scatter(
        [steps[best_idx]],
        [impr_median_test[best_idx]],
        color="C2",
        zorder=5,
        s=80,
        marker="*",
    )
    ax_main.annotate(
        f"best: {impr_median_test[best_idx]:.1f}%\nstep {steps[best_idx]}",
        xy=(steps[best_idx], impr_median_test[best_idx]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=9,
        color="C2",
        arrowprops=dict(arrowstyle="->", color="C2", lw=1),
    )

ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax_main.set_xlabel("Compression Step", fontsize=12)
ax_main.set_ylabel("RMSE Improvement (%, higher = better)", fontsize=12)
ax_main.set_title("High-Quality RMSE vs Compression Steps", fontsize=12)
ax_main.grid(True, alpha=0.4)
ax_main.legend(
    fontsize=9,
    loc="upper left",
    ncol=2,
    framealpha=0.9,
)

# -- weights
if show_individual:
    for r in runs_to_plot:
        ax_w.plot(
            weight_log[r],
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
            value_log[r],
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
        n_dims_r = (weight_log[r] > 0.001).sum(axis=1)
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
plt.savefig(path.replace(".npz", "_plot.png"), dpi=150, bbox_inches="tight")
print(f"Saved {path.replace('.npz', '_plot.png')}")
plt.show()

# %%
