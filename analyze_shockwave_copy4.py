"""Post-run analysis replicating shockwave-260122test-Copy4.ipynb cells 5-6.

Loads the detected wavefronts (util.load_wavefronts_filtered), sums them,
prints the notebook's diagnostics, plots wf[0], and writes the profile to a
text file exactly like the notebook's cell 6 ('3D_before_inverse_version').
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

repo = Path("/mnt/d/rave-sim-main/rave-sim-main")
sys.path.pop(0)  # avoid shadowing the installed nist_lookup package
sys.path.insert(0, str(repo / "big-wave"))

import config  # noqa: E402
import util  # noqa: E402

sim_path = Path("/mnt/d/rave-sim-main/rave-sim-main/output/2026/08/20260815_113841601813__agentrun")
out_txt = sim_path / "3D_before_inverse_version"
out_png = sim_path / "wf0_lineout.png"

# --- notebook cell 5 tail ---
wavefronts = util.load_wavefronts_filtered(sim_path, x_range=(-100 * 1e-6, 100 * 1e-6))
print("nr sources loaded:", len(wavefronts))
wavef = [result[0] for result in wavefronts]
wf = np.sum(wavef, axis=0)
print("nr phase steps:", wf.shape[0])
print("nr detector pixels:", wf.shape[1])
sp = config.load(sim_path / "config.yaml")["sim_params"]
detector_x = util.detector_x_vector(sp["detector_size"], sp["detector_pixel_size_x"])
print("detector_x len:", len(detector_x), "first/last:", detector_x[0], detector_x[-1])

# Plot like the notebook: wf[0] vs index, plus a labeled x-axis variant
n = wf.shape[1]
x_centered = (np.arange(n) - n // 2) * sp["detector_pixel_size_x"]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(wf[0], lw=1.2)
ax.set_xlabel("detector pixel index")
ax.set_ylabel("intensity")
ax.set_title("shockwave-260122test-Copy4 | wf[0] (as in notebook)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print("saved:", out_png)

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(x_centered * 1e3, wf[0], lw=1.2)
ax2.set_xlabel("detector x (mm)")
ax2.set_ylabel("intensity")
ax2.set_title("shockwave-260122test-Copy4 | wf[0] vs detector x")
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(sim_path / "wf0_lineout_vs_x.png", dpi=150)

# --- notebook cell 6 ---
with open(out_txt, "w") as f:
    for v in wf[0]:
        f.write(f"{v}\n")
print("wrote:", out_txt)

print("wf[0] =", wf[0])
