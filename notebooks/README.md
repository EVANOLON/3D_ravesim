# RAVE-SIM notebooks

The notebooks in this directory are retained as examples and research records.
They are not all supported release entry points. Automated correctness checks
belong in `tests/`; large paper-scale inputs and generated wave fields belong in
external data storage.

## Recommended starting points

| Entry | Purpose | Data requirement | Status |
| --- | --- | --- | --- |
| `../examples/minimal_xpci/` | Small, portable 2D XPCI run | Generated in memory | Supported quick start |
| `test_2d_simulation/test_2d.ipynb` | 2D propagation and sample validation | Generated in memory | Candidate public notebook |
| `test_plasma/` | `PlasmaSample` demonstrations | Some notebooks need local arrays | Candidate; dependencies need review |
| `3D_test_notebooks/` | Voxel sample and material studies | Local grids and a CUDA build | Research examples; select one |
| `capsule_test_0809/` | Compact capsule workflow | Generated/local shell grid | Research example; under review |
| `euler-example.ipynb`, `tl3g-usage-example.ipynb` | Continuity with the original ETH workflows | Site-specific paths/builds | Legacy reference |
| `Crack_simulation.ipynb`, `Dual_phase_comparison_simulation.ipynb`, `Thresholding_Simulation.ipynb` | Analysis and detector studies | Local data and optional packages | Research reference |

Start with the minimal example unless a notebook-specific workflow is required.

## Local data safety

This working tree may contain ignored `.npy`, `.h5`, text output, figures,
checkpoints, and historical experiment directories. They can be many gigabytes
even though the Git-tracked notebooks are small. Do not run `git clean -f` or
`git clean -fX` against this directory without reviewing the preview and backing
up research data.

Safe inspection commands:

```bash
python tools/notebook_audit.py
python tools/notebook_audit.py --all
git status --ignored --short notebooks
git clean -ndX notebooks
```

The last command is preview-only; keep the `-n` option.

## Publication criteria

A notebook is ready to be a supported public example only when it:

1. discovers the repository without a machine-specific absolute path;
2. either generates its small inputs or documents how to obtain versioned data;
3. runs from a clean environment with the declared dependencies;
4. completes in a stated, practical amount of time;
5. writes generated artifacts under the ignored top-level `output/` directory;
6. contains concise expected results and no large embedded cell outputs.

Historical capsule experiment directories removed from Git are intentionally not
part of the release. A local ignored copy is data to review or archive, not a
candidate to re-add.
