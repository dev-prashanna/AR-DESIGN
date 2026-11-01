"""Method diagram, not an experimental result."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.set(xlim=(0, 10), ylim=(0, 2.3))
ax.axis("off")
items = [
    (0.05, 1.1, 1.5, "27 proprioceptive\nobservations"),
    (1.9, 1.1, 1.7, "Normalization\nactor MLP"),
    (4.0, 1.1, 1.4, "Gaussian u\ntanh + limits"),
    (5.8, 1.1, 1.6, "PD actuator\n500 Hz"),
    (7.8, 1.1, 2.0, "MuJoCo Warp\nN independent worlds"),
]
for x, y, width, label in items:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, 0.7, boxstyle="round,pad=.04", facecolor="#e4f0f2", edgecolor="#32666f"
        )
    )
    ax.text(x + width / 2, y + 0.35, label, ha="center", va="center", fontsize=9)
for end, start in [(1.55, 1.9), (3.6, 4.0), (5.4, 5.8), (7.4, 7.8)]:
    ax.annotate("", xy=(start, 1.45), xytext=(end, 1.45), arrowprops={"arrowstyle": "->"})
ax.annotate(
    "",
    xy=(0.8, 1.05),
    xytext=(8.8, 1.05),
    arrowprops={"arrowstyle": "->", "connectionstyle": "bar,fraction=-.08"},
)
ax.text(
    4.7,
    0.24,
    "50 Hz policy; N x 24 rollout transitions -> shared PPO actor + critic update",
    ha="center",
    fontsize=9,
)
ax.text(
    2.75,
    2.08,
    "Critic uses the same 27 fields; terminal values are captured before reset",
    ha="center",
    fontsize=8,
)
fig.tight_layout()
fig.savefig(root / "results/figures/control_loop.pdf", bbox_inches="tight")
fig.savefig(root / "results/figures/control_loop.png", dpi=160, bbox_inches="tight")
