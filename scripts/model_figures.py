"""Plots only measured model-validation data. Run after biped validate-model."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "docs/model_audit.json").read_text())
trace = data["standing_trace"]
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot([x[0] for x in trace], [1000 * x[1] for x in trace], color="#258593")
ax.set(
    xlabel="Physical time (s)",
    ylabel="Torso height (mm)",
    title="Floating-base PD standing diagnostic",
)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(root / "results/figures/standing.png", dpi=180)
plt.close(fig)
