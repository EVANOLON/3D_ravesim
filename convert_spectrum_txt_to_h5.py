"""
Convert spectrum data from a text file to HDF5 format compatible with
spectrum_25keV.h5.

The HDF5 file must contain two datasets:
  - "energy" : energy values in eV
  - "pdf"    : normalized probability density (sum = 1)

Usage:
    python convert_spectrum_txt_to_h5.py <input.txt> <output.h5>

Example:
    python convert_spectrum_txt_to_h5.py spectrum/spectrum_microX.txt spectrum/spectrum_microX.h5
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def convert_txt_to_h5(input_path: Path, output_path: Path,
                      skiprows: int = 0, delimiter: str = None,
                      energy_col: int = 0, intensity_col: int = 1,
                      energy_unit: str = "eV") -> None:
    """
    Read a two-column spectrum text file and write HDF5.

    Parameters
    ----------
    input_path : Path
        Path to the input text file.
    output_path : Path
        Path for the output .h5 file.
    skiprows : int
        Number of header rows to skip.
    delimiter : str or None
        Column delimiter (None = auto-detect whitespace).
    energy_col : int
        Column index for energy values.
    intensity_col : int
        Column index for intensity values.
    energy_unit : str
        Unit of the energy column: "eV" or "keV".
    """
    # 1. Read the text file (open with UTF-8 for Windows compatibility)
    print(f"Reading: {input_path}")
    with open(input_path, encoding="utf-8", errors="replace") as file_handle:
        data = np.loadtxt(file_handle, skiprows=skiprows, delimiter=delimiter)

    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns, got shape {data.shape}. "
            "Check skiprows/delimiter."
        )

    energy = data[:, energy_col]
    intensity = data[:, intensity_col]

    # 2. Convert energy units if needed
    if energy_unit.lower() == "kev":
        energy = energy * 1000  # keV -> eV
        print("Converted energy from keV to eV")
    elif energy_unit.lower() != "ev":
        print(f"Warning: unknown energy unit '{energy_unit}', assuming eV")

    # 3. Ensure intensities are non-negative
    if np.any(intensity < 0):
        print("Warning: negative intensities found, clipping to 0")
        intensity = np.clip(intensity, 0, None)

    # 4. Normalize to probability density
    pdf = intensity / np.sum(intensity)

    # 5. Write HDF5
    print(f"Writing: {output_path}")
    print(f"  energy: {len(energy)} points, range [{energy[0]:.1f}, {energy[-1]:.1f}] eV")
    print(f"  pdf:    sum = {np.sum(pdf):.6f}")

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("energy", data=energy)
        h5.create_dataset("pdf", data=pdf)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python convert_spectrum_txt_to_h5.py <input.txt> [output.h5]")
        print("")
        print("If output.h5 is omitted, replaces .txt extension with .h5")
        print("")
        print("Examples:")
        print("  # Spekpy text format (18 header lines, keV tab-separated)")
        print("  python convert_spectrum_txt_to_h5.py spectrum/spectrum_microX.txt")
        print("")
        print("  # Custom format")
        print("  python convert_spectrum_txt_to_h5.py my_spectrum.txt my_spectrum.h5")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_suffix(".h5")

    # Auto-detect: Spekpy files have 18 header lines
    # (17 comment lines + 1 blank + column headers at line 18)
    with open(input_path, encoding="utf-8", errors="replace") as f:
        first_lines = [f.readline() for _ in range(20)]

    spekpy_hint = any("# Spekpy" in l for l in first_lines)
    n_headers = sum(1 for l in first_lines if l.startswith("#")) + 1  # +1 for column header

    if spekpy_hint:
        print(f"Detected Spekpy format ({n_headers} header lines)")
        convert_txt_to_h5(input_path, output_path,
                          skiprows=n_headers, delimiter="\t",
                          energy_col=0, intensity_col=1,
                          energy_unit="keV")
    else:
        convert_txt_to_h5(input_path, output_path)
