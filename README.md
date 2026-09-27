# Learning Stochastic Behaviour from Aggregate Data

Clean research-code repository accompanying:

**Shaojun Ma, Shu Liu, Hongyuan Zha, Haomin Zhou**  
*Learning Stochastic Behaviour from Aggregate Data*  
Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 2021.

Paper: https://proceedings.mlr.press/v139/ma21c.html

## Overview

This project learns hidden stochastic dynamics when only **aggregate snapshots**
are observed.

Instead of tracking the same individual through time, the data consist of
independent samples from the population distribution at several time points:

```text
X_0 ~ p(x,t_0)
X_1 ~ p(x,t_1)
...
X_J ~ p(x,t_J)
```

Individual identities and complete trajectories are not required.

The latent dynamics are modeled by the stochastic differential equation

```text
dX_t = g_theta(X_t) dt + sigma dW_t
```

where the unknown drift field `g_theta` is represented by a neural network.

The method combines:

- the **weak form of the Fokker-Planck equation**,
- **Wasserstein-1 duality**,
- WGAN-style 1-Lipschitz test functions,
- sample-based Euler-Maruyama simulation,
- alternating optimization of the drift and Wasserstein critics.

The key point is that the Fokker-Planck PDE is never discretized on a spatial
grid.

---

## Weak Fokker-Planck Operator

For a smooth test function `f`, define

```text
L_g f(x)
    =
g_theta(x) . grad f(x)
+
0.5 * sigma^2 * Laplacian f(x)
```

The weak Fokker-Planck identity implies that, over a time interval,

```text
E[f(X_t)] - E[f(X_0)]
    =
Integral E[L_g f(X_s)] ds
```

The code approximates this time integral with a trapezoidal rule evaluated on
sampled states.

For an observed aggregate snapshot at time `t_j`, the resulting discrepancy is

```text
D_j(g,f_j)
    =
E_data_tj[f_j]
-
E_data_t0[f_j]
-
Integral_0^tj E[L_g f_j] dt
```

Each `f_j` is constrained to be approximately 1-Lipschitz using spectral
normalization.

Training alternates between:

```text
critic step:
    maximize D_j(g, f_j)

drift step:
    minimize sum_j D_j(g, f_j)
```

---

## Repository Structure

```text
learning_stochastic_behavior_aggregate_data/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── synthetic_2d.json
│   └── paper_reference.json
├── src/
│   └── aggregate_dynamics/
│       ├── __init__.py
│       ├── models.py
│       ├── weak_fpe.py
│       ├── simulation.py
│       ├── trainer.py
│       ├── data.py
│       └── metrics.py
├── experiments/
│   ├── synthetic_2d.py
│   ├── gene_expression_template.py
│   └── trading_volume_template.py
├── docs/
│   └── VERSION_AUDIT.md
├── data/
│   └── README.md
└── figures/
```

---

## Synthetic Experiments

The paper studies three synthetic systems, including linear dynamics,
nonlinear mixture dynamics, and a Van der Pol-type oscillator.

For the reported synthetic experiments:

- drift network `g`: 1 hidden layer,
- critic network `f`: 3 hidden layers,
- hidden width: 32,
- activation: `tanh`,
- optimizer: Adam,
- learning rate: `1e-4`,
- samples per observation time: 2000,
- train/test split: 1200 / 800,
- time step: `0.01`,
- spectral normalization on the critic.

A compact 2-D runnable example is included:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python experiments/synthetic_2d.py
```

The default configuration intentionally uses fewer iterations than the full
historical research run.

---

## Single-Cell RNA-seq Experiment

The paper uses aggregate single-cell RNA-seq observations at:

```text
D0, D2, D4, D7
```

with no cell identity linking one observation time to the next.

Ten gene markers are used to form a 10-dimensional aggregate state.

The paper evaluates two prediction tasks:

```text
train on D0, D4, D7 -> predict D2
train on D0, D2, D4 -> predict D7
```

The original data files are not redistributed here.

`experiments/gene_expression_template.py` documents the expected `.npy` inputs
and the paper-style 10-D training interface.

---

## Intraday Trading-Volume Experiment

The paper also applies the method to historical JPM traded volume.

The reported data:

- January 2018 to January 2020,
- 78 observations per trading day,
- one observation every five minutes,
- natural-log volume preprocessing.

The original Bloomberg-derived data are not redistributed.

`experiments/trading_volume_template.py` contains the cleaned data-layout logic
and shows how aggregate snapshots are extracted from a user-provided CSV.

---

## Historical Code Consolidation

The uploaded research code contained four large experimental files:

- `FPP_syn.py` — primary 2-D synthetic implementation,
- `FPP.py` — larger exploratory synthetic variant with extensive figure/debug code,
- `FPP_gene.py` — 10-D gene-expression branch,
- `FPP_volume_prediction-sn.ipynb` — intraday JPM volume application.

All four repeatedly define:

- spectral normalization,
- drift and critic networks,
- first/second derivatives,
- weak FPE operator,
- trapezoidal weak-FPE integral,
- alternating critic / drift updates.

Those common components are now centralized in `src/aggregate_dynamics/`.

The cleaned repository does **not** copy thousands of lines of repeated plotting
and commented-out code.

See `docs/VERSION_AUDIT.md` for details.

---

## Reproducibility Notes

This repository is a consolidated and readable implementation of the method.
It is not a claim that every historical plotting cell or every external-data
experiment can be reproduced without the original data.

External datasets are intentionally excluded.

The synthetic core is self-contained.

---

## Citation

```bibtex
@inproceedings{ma2021learning,
  title={Learning Stochastic Behaviour from Aggregate Data},
  author={Ma, Shaojun and Liu, Shu and Zha, Hongyuan and Zhou, Haomin},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  pages={7252--7262},
  year={2021},
  organization={PMLR}
}
```
