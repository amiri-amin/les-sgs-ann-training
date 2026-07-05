# LES SGS-ANN Training — Weights & Biases for the Fortran Closure

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00.svg)](https://www.tensorflow.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-notebook-f37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The **training and data-preparation pipeline** that produces the neural-network **weights and
biases** consumed by the Fortran LES subgrid-scale (SGS) stress closure.

Starting from a high-fidelity LES flow field (HDF5), this notebook applies the LES filtering
operator to compute the filtered velocities and the **filtered SGS stress tensor τ<sub>ij</sub>**,
assembles the training dataset, trains an artificial neural network (ANN) in Keras, and
exports the learned parameters for use inside the flow solver.

> **Companion project:** the exported weights/biases are read at run time by the Fortran
> closure module (`ANNClosures_m`) in the **ML-Augmented SGS Closure for LES (Fortran 90)**
> repository. This repo is the *training* side; that repo is the *inference* side.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [The physics](#the-physics)
- [The network](#the-network)
- [Repository layout](#repository-layout)
- [Inputs &amp; outputs](#inputs--outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Handing off to Fortran](#handing-off-to-fortran)
- [Citation](#citation)
- [License](#license)

---

## Pipeline overview

```
  LES field (HDF5)        filtering              features / labels          ANN            export
 ┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌───────────┐   ┌──────────────┐
 │ U, V, W + grid │──▶│ 3×3×3 LES filter │──▶│ X = [ū, v̄, w̄]     │──▶│ 3→40→40→6│──▶│ weights.npy  │
 │  Fld.090.h5    │   │ volume-weighted  │   │ Y = τ_ij (6 comp.) │   │ ReLU+lin. │   │ biases.npy   │
 └────────────────┘   └──────────────────┘   └────────────────────┘   └───────────┘   └──────────────┘
```

1. **Load** velocities `U, V, W` and grid `X, Y, Z` from `Fld.090.h5` (`h5py`).
2. **Build the filter** — a 3×3×3 trilinear LES kernel, plus central-difference kernels used
   to compute cell volumes.
3. **Filter** — volume-weighted (Favre-style) filtering to obtain the filtered velocities
   FU1, FU2, FU3 (ū, v̄, w̄) via `scipy.ndimage.correlate`.
4. **SGS stress** — compute the filtered stress tensor
   `τ_ij = filter(U_i U_j) − ū_i ū_j` for the 6 independent components (FT11 … FT33).
5. **Assemble** — flatten to `X_input = [ū, v̄, w̄]` (min–max normalized) and
   `Y_output = [τ11, τ12, τ13, τ22, τ23, τ33]`; split 90 % train / 10 % test.
6. **Train** the ANN and monitor loss / percentage error.
7. **Export** the trained weights and biases as `.npy` for the Fortran solver.

---

## The physics

In LES the resolved-scale momentum equations are unclosed because of the subgrid-scale stress

```
τ_ij = (u_i u_j)_filtered  −  ū_i ū_j
```

Here the filtering is performed explicitly with a discrete 3×3×3 kernel and volume weighting,
so that both sides of the definition are computed directly from the LES field. The resulting
(ū, v̄, w̄) → τ<sub>ij</sub> pairs form the supervised training set for the ANN, which then
serves as a data-driven replacement for an analytical closure inside the solver.

---

## The network

A Keras functional model:

| Layer | Units | Activation |
|-------|:-----:|------------|
| Input | 3 (ū, v̄, w̄) | — |
| Hidden 1 | 40 | ReLU |
| Hidden 2 | 40 | ReLU |
| Output | 6 (τ<sub>ij</sub>) | Linear |

- **Optimizer:** Adamax (`lr = 1e-3`)
- **Loss:** mean squared error
- **Training:** 20 epochs, batch size 10 000, 90/10 train/test split
- **Diagnostics:** training-vs-validation loss curve and a percentage-error plot normalized by
  the zero-prediction MSE baseline

This 3→40→40→6 architecture matches the network expected by the Fortran closure.

---

## Repository layout

```
les-sgs-ann-training/
├── ANN_NEW_21_FTij_Hyperparameters_01.ipynb   # Full filtering → training → export pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

At run time the notebook expects/creates:

```
Fld.090.h5        # (input, not included) LES flow field: FlowData/{U,V,W}, Grid/{X,Y,Z}
weights/          # (output) all_weights.npy
biases/           # (output) all_biases.npy
```

---

## Inputs & outputs

**Input** — an HDF5 LES snapshot with this structure:

```
Fld.090.h5
├── FlowData/
│   ├── U, V, W        # velocity component fields (3D)
└── Grid/
    └── X, Y, Z        # node coordinates (3D)
```

**Output** — the trained parameters:

| File | Contents |
|------|----------|
| `weights/all_weights.npy` | List of per-layer weight matrices (hidden 1, hidden 2, output) |
| `biases/all_biases.npy` | List of per-layer bias vectors |

---

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

1. Place your LES snapshot `Fld.090.h5` in the working directory.
2. Create the output folders: `mkdir -p weights biases`.
3. Launch Jupyter and run the notebook top to bottom:

```bash
jupyter notebook ANN_NEW_21_FTij_Hyperparameters_01.ipynb
```

The final cells save `weights/all_weights.npy` and `biases/all_biases.npy`.

---

## Handing off to Fortran

The Fortran closure reads plain-text `weights.data` / `biases.data` with a small header
(`nLayers  maxNodes`). Convert the exported `.npy` arrays into that layout, then point the
solver's `inPath` at the resulting files. See the companion **Fortran 90** repository's
*Data-file formats* section for the exact header/row convention.

---

## Citation

If you use this pipeline, please credit this repository:

```bibtex
@software{amiri_les_sgs_ann_training,
  author = {Amiri, Amin},
  title  = {LES SGS-ANN Training: Weights and Biases for a Fortran LES Closure},
  year   = {2022},
  url    = {https://github.com/amiri-amin/les-sgs-ann-training}
}
```

## License

Released under the [MIT License](LICENSE) © Amin Amiri.
