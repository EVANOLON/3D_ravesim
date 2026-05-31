"""Analyze W/C capsule absorption from detected.npy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MATERIALS = {
    "W": {
        "output_dir": Path(r"D:\rave-sim-main\rave-sim-main\output\2026\05\20260517_023258291382\00000000"),
        "density": 19.35, "label": "Tungsten (W)", "color": "tab:red", "Z": 74,
    },
    "C": {
        "output_dir": Path(r"D:\rave-sim-main\rave-sim-main\output\2026\05\20260517_021803941821\00000000"),
        "density": 1.5, "label": "Carbon (C)", "color": "tab:blue", "Z": 6,
    },
}
NX, NY = 4250, 4250
DET_PIXEL = 2e-7  # 200 nm
SPHERE_R_UM = 500
MAG = 4.0
GEOM_EDGE_UM = SPHERE_R_UM * 2 * MAG  # ~4000 um

# 10 keV X-ray mass attenuation coefficients (approx from NIST)
MASS_ATTEN = {"W": 230, "C": 5, "H": 0.4}  # cm^2/g


def radial_profile(img, cx=None, cy=None, step=2):
    if cx is None: cx = img.shape[1] // 2
    if cy is None: cy = img.shape[0] // 2
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.int32)
    r_max = int(rr.max()) + 1
    w = img.ravel()
    r = rr.ravel()
    return np.arange(r_max), np.bincount(r, weights=w) / np.bincount(r)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, (key, cfg) in enumerate(MATERIALS.items()):
        det = np.load(cfg["output_dir"] / "detected.npy").squeeze().astype(np.float64)

        # Statistics
        cy, cx = NY // 2, NX // 2
        center = det[cy-50:cy+50, cx-50:cx+50]
        edge_corners = np.r_[det[:50, :50], det[:50, -50:],
                             det[-50:, :50], det[-50:, -50:]]
        edge_m = edge_corners.mean()
        center_m = center.mean()

        print(f"[{key}] {cfg['label']}")
        print(f"  Shape: {det.shape}")
        print(f"  Range: [{det.min():.3e}, {det.max():.3e}]")
        print(f"  Mean:  {det.mean():.3e}")
        print(f"  Center(100x100) mean: {center_m:.3e}")
        print(f"  Edge(4x50x50)   mean: {edge_m:.3e}")
        print(f"  Center/Edge: {center_m/edge_m:.4f}")

        # Absorption estimate through sphere center (200 um path)
        mu = MASS_ATTEN[key] * cfg["density"] * 100  # 100 um path -> absorption factor unit conversion
        path_um = 200  # through full sphere diameter
        mu_cm = MASS_ATTEN[key] * cfg["density"]  # cm^-1
        T = np.exp(-mu_cm * path_um * 1e-4)  # convert um to cm
        print(f"  mu={mu_cm:.1f} cm^-1, T(200um)={T:.2%}")
        print()

        # Image
        ax_img = axes[0, i]
        extent = [0, NX*DET_PIXEL*1e6, 0, NY*DET_PIXEL*1e6]
        im = ax_img.imshow(det, cmap="inferno", origin="lower", extent=extent)
        ax_img.set_title(f"{cfg['label']}  (Z={cfg['Z']}, rho={cfg['density']})")
        ax_img.set_xlabel("x (um)"); ax_img.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax_img, fraction=0.046)

        # Radial profile
        ax_rad = axes[1, i]
        r_px, radial_m = radial_profile(det)
        r_um = r_px * DET_PIXEL * 1e6
        ax_rad.plot(r_um, radial_m, color=cfg["color"], lw=1)
        ax_rad.axvline(SPHERE_R_UM * 2 * MAG, ls="--", color="gray", alpha=0.5,
                       label=f"Geom. edge ({SPHERE_R_UM*2*MAG:.0f} um)")
        ax_rad.set_xlabel("Radius (um)"); ax_rad.set_ylabel("Intensity")
        ax_rad.set_title("Radial profile")
        ax_rad.set_yscale("log")
        ax_rad.legend(fontsize=8)
        ax_rad.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("absorption_analysis.png", dpi=150)
    print("Saved absorption_analysis.png")

    # Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for key, cfg in MATERIALS.items():
        det = np.load(cfg["output_dir"] / "detected.npy").squeeze().astype(np.float64)
        r_px, radial_m = radial_profile(det)
        r_um = r_px * DET_PIXEL * 1e6
        ax2.plot(r_um, radial_m / radial_m[-500:].mean(), color=cfg["color"],
                 lw=1.5, label=cfg["label"])
    ax2.axvline(SPHERE_R_UM * 2 * MAG, ls="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Radius (um)"); ax2.set_ylabel("Normalized intensity")
    ax2.set_title("Normalized Radial Profile")
    ax2.set_yscale("log"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig("absorption_comparison.png", dpi=150)
    print("Saved absorption_comparison.png")

    print("\n--- Expected ---")
    print("W: very high absorption -> deep central shadow, Poisson spot expected but dim")
    print("C: weak absorption -> visible Fresnel fringes + Poisson spot (center bright)")


if __name__ == "__main__":
    main()
