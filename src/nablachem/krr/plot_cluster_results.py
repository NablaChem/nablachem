#!/usr/bin/env python3
# %% imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# %% load

path = next(
    (a for a in sys.argv[1:] if a.endswith(".npz")),
    "/Users/ali/second_project_data/results/out.npz",
)
d = np.load(path)

steps = d["rmse_steps"]  # (n_eval_steps,)
test_errors = d["test_errors"]  # (n_runs, n_eval_steps)
val_errors = d["val_errors"]  # (n_runs, n_eval_steps)
value_log = d["value_log"]  # (n_runs, n_steps)
weight_log = d["weight_log"]  # (n_runs, n_steps, 40)

mean_test = d["mean_test_errors"]  # (n_eval_steps,)
mean_val = d["mean_val_errors"]  # (n_eval_steps,)
mean_value = d["mean_value_log"]  # (n_steps,)
mean_weights = d["mean_weight_log"]  # (n_steps, 40)

n_runs = test_errors.shape[0]

# %% derived

ref_test = mean_test[0]
ref_val = mean_val[0]
impr_test = (ref_test - mean_test) / ref_test * 100
impr_val = (ref_val - mean_val) / ref_val * 100

running_avg = np.cumsum(mean_value) / (np.arange(len(mean_value)) + 1)
n_dims = (mean_weights > 0.001).sum(axis=1)

# %% figure

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.3)
ax_main = fig.add_subplot(gs[:, 0])
ax_w = fig.add_subplot(gs[0, 1])
ax_v = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[2, 1])

fig.suptitle(
    f"Cluster results — {path}  ({n_runs} run{'s' if n_runs > 1 else ''})",
    fontsize=14,
    fontweight="bold",
)

# -- per-run shading
for r in range(n_runs):
    impr_r = (ref_test - test_errors[r]) / ref_test * 100
    ax_main.plot(steps, impr_r, color="C0", alpha=0.2, linewidth=0.8)

# -- median line
ax_main.plot(steps, impr_test, marker="o", color="C0", label="Test RMSE (median)")
ax_main.plot(
    steps,
    impr_val,
    marker="o",
    color="C0",
    linestyle="--",
    alpha=0.5,
    label="Val RMSE (median)",
)

for x, y in zip(steps, impr_test):
    ax_main.annotate(
        f"{y:.1f}%",
        xy=(x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="C0",
    )

ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax_main.set_xlabel("Compression Step", fontsize=12)
ax_main.set_ylabel("RMSE Improvement (%)", fontsize=12)
ax_main.set_title("High-Quality RMSE vs Compression Steps", fontsize=12)
ax_main.invert_yaxis()
ax_main.grid(True, alpha=0.4)
ax_main.legend(
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    framealpha=0.9,
    borderaxespad=0,
)

# -- weights
for r in range(n_runs):
    ax_w.plot(weight_log[r], alpha=0.1, linewidth=0.4, color="steelblue")
ax_w.plot(mean_weights, alpha=0.6, linewidth=0.8, color="steelblue")
ax_w.set_title("Feature Weights (median)")
ax_w.set_ylabel("Weight")
ax_w.set_xlabel("Step")
ax_w.grid(True, alpha=0.3)

# -- LQ value log
ax_v.semilogy(mean_value, color="C3", alpha=0.4, linewidth=0.8, label="per-step")
ax_v.semilogy(running_avg, color="C1", linewidth=1.5, label="running avg")
ax_v.set_title("Low-Quality Val Error")
ax_v.set_ylabel("RMSE (log)")
ax_v.set_xlabel("Step")
ax_v.legend(fontsize=8)
ax_v.grid(True, alpha=0.3)

# -- active dims
ax_d.plot(n_dims, color="C2")
ax_d.set_title("Active Dimensions")
ax_d.set_ylabel("Count")
ax_d.set_xlabel("Step")
ax_d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path.replace(".npz", "_plot.png"), dpi=150, bbox_inches="tight")
print(f"Saved {path.replace('.npz', '_plot.png')}")
plt.show()

# %%
