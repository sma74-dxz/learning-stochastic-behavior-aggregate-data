# Version Audit

This document records how the uploaded historical code was consolidated.

## Uploaded files

| Original file | Role | Decision |
|---|---|---|
| `FPP_syn.py` | Compact 2-D synthetic weak-FPE/Wasserstein implementation | **Primary source for the synthetic core** |
| `FPP.py` | Large exploratory synthetic script containing repeated model definitions, diagnostics, plotting, and alternative dynamics | Historical / experiment-development source |
| `FPP_gene.py` | 10-D gene-expression branch using D0/D2/D4/D7 snapshots | Application source; converted to an external-data template |
| `FPP_volume_prediction-sn.ipynb` | JPM intraday log-volume prediction with aggregate snapshots | Application source; converted to an external-data template |

## Common code identified across versions

The historical files independently reimplement the same building blocks:

1. spectral normalization for the Wasserstein test function,
2. neural drift `g`,
3. neural test function / critic `f`,
4. automatic differentiation of first and second derivatives,
5. the weak Fokker-Planck generator

```text
g . grad(f) + 0.5 * sigma^2 * Delta(f)
```

6. trapezoidal integration of the weak FPE term,
7. alternating inner critic updates and outer drift updates,
8. Euler-Maruyama sample propagation.

These are now centralized under `src/aggregate_dynamics/`.

## Important cleanup decisions

- Modern PyTorch spectral normalization replaces the custom duplicated
  `SpectralNorm` implementation.
- Deprecated `torch.autograd.Variable` usage is removed.
- Repeated NumPy-to-Torch-to-NumPy conversions inside training loops are removed.
- Figure-generation code is separated from the mathematical core.
- Hard-coded local paths such as `/tmp/D0.npy` are removed.
- External RNA-seq and Bloomberg-derived JPM data are not redistributed.
- The clean implementation uses dimension-generic tensor operations instead of
  manually splitting every coordinate into separate arrays.
- The gene-expression historical script contains experimental code that is not
  internally dimension-consistent in every branch (for example, a 10-D state
  branch also contains an older low-dimensional output declaration). The public
  version therefore keeps the 10-D application as a clean template rather than
  copying that branch verbatim.

## Paper alignment

The cleaned synthetic implementation follows the method in the paper:

- time-independent drift `g(x)`,
- constant diffusion `sigma`,
- weak Fokker-Planck operator,
- Wasserstein-1 critics,
- spectral normalization,
- trapezoidal time integration,
- alternating critic/drift training.

Extensions discussed in the paper, such as learning a state-dependent diffusion,
are intentionally kept outside the core implementation.
