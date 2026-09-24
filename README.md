# Anchored graph neural network model

This repository provides a public implementation of the anchored grain graph
framework for predicting hardening and Lankford coefficients in
BCC polycrystals. It contains the graph builder, model, joint loss,
normalization, Sachs/Taylor reference calculation, training, inference, evaluation, and the fixed condition split
used in the study.

Release 2.1.0 uses the anchored model and training procedure described in the
current manuscript: two message-passing layers, 48 latent grain features,
51,623 trainable parameters, and an initial reference mixing weight of 0.5.
AdamW uses a constant learning rate of 0.002 and weight decay of 0.0001,
with batches of 16, at most 160 epochs, and early stopping after 25 epochs
without validation improvement. The retained checkpoint minimizes the sum
of hardening and Lankford mean relative errors. No moving-average weights
or learning-rate schedule are used.

The objective is `L_H + 2 L_r + 0.15 L_aux`, where `L_H` combines the
standardized log-radius error with 0.15 times the hardening-parameter error.
`L_aux` averages the stress-descriptor and tensile-flow-stress errors.
The directional correction and reference mixing weight are learned without
separate penalties. All response errors use SmoothL1; its transition scale
is 0.5 for standardized log-r errors and 1 for the other terms.

A complete twenty-condition example is included under `examples/`, with fourteen training, three validation and three test textures. It contains microstructure inputs, graph arrays, postprocessed CPFEM targets, anchor estimates, a fresh example-trained checkpoint, and reproducible results. The full research dataset and the paper's trained weights are not included.

Start with [the executable example](examples/README.md):

```bash
python -m pip install -r examples/requirements.txt
python examples/run_example.py --seed 7
```

Normalization statistics are fitted on the training subset. Runs default to
seed 7; the manuscript uses independent seeds 7, 8 and 9 and averages their
predictions. The bundled example contains only a single seed-7 run. It uses
the same anchored architecture, loss and training settings but fits normalization and weights on
its fourteen training textures, so its scores are not the full-study scores.
Solver models and simulation-project files are not distributed.

## Repository contents

- `anchored_gnn/graph.py`: periodic grain graph construction
- `anchored_gnn/anchors.py`: Sachs/Taylor references from grain orientations and volumes
- `anchored_gnn/model.py`: anchored grain graph network
- `anchored_gnn/workflow.py`: batching, training-set normalization, joint loss,
  Hockett-Sherby decoding, and metrics
- `anchored_gnn/train.py`: training and validation checkpoint selection
- `anchored_gnn/predict.py`: inference
- `anchored_gnn/evaluate.py`: split-specific evaluation
- `configs/split.csv`: fixed train/validation/test assignment
- `configs/training.json`: model and training settings
- `tests/smoke_test.py`: small runnable implementation check

The fixed split records which condition identifiers belong to each subset. It
keeps model fitting, checkpoint selection, and final testing separated and
makes comparisons use the same held-out conditions.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

## Input schemas

The bundled example follows the schema below. For your own data, supply one graph file per condition as
`GRAPHS/<condition_id>.npz`. Each file must contain:

| Array | Shape | Meaning |
|---|---:|---|
| `x` | `(grains, 220)` | volume, centroid, and Schmid node channels |
| `edge_index` | `(2, edges)` | directed source and target indices |
| `edge_attr` | `(edges, 6)` | boundary and neighbour-pair channels |
| `volume` | `(grains,)` | grain volume fractions |

The target bundle is an NPZ file with the following arrays:

| Array | Shape |
|---|---:|
| `condition_ids` | `(conditions,)` |
| `hardening_parameters` | `(conditions, 4, 9)` |
| `hardening_radius` | `(conditions, 7, 9)` |
| `hardening_strain` | `(conditions, 7, 9)` |
| `r_values` | `(conditions, 3)` |
| `anchor_lower`, `anchor_upper` | `(conditions, 3)` each |
| `stress_aux` | `(conditions, 126)` |
| `flow_aux` | `(conditions, 9)` |

All feature and target normalization statistics are fitted from the training
subset only and stored in the checkpoint. The two anchor arrays contain the
independently calculated Sachs and Taylor reference estimates used by the Lankford decoder.

## Build a graph

The graph builder reads a three-dimensional integer grain map, a grain-indexed
array of Bunge Euler angles in radians, and a 24-row BCC slip-system table. It
does not read simulation-project files.

```bash
python -m anchored_gnn.graph \
  --feature-ids INPUT/feature_ids.npy \
  --eulers INPUT/grain_eulers.npy \
  --slip-systems INPUT/slip_systems.txt \
  --output GRAPHS/TS001.npz
```

## Train, infer, and evaluate

First calculate reference estimates for new microstructures, using the same
NPZ schema as `examples/microstructures/`:

```bash
python -m anchored_gnn.anchors --microstructures INPUT/microstructures --slip-systems examples/slip_systems.txt --output INPUT/anchors.npz
```

The calculation reads only grain geometry, orientations and the slip-system
table. Sachs uses uniform stress and a slip power-law exponent of 20; Taylor
uses uniform strain with exponent 8 and solves for the lateral strain ratio
under the macroscopic uniaxial-stress condition. These are the established
reference-calculation settings, distinct from the CPFEM constitutive model.
Orientations and volume fractions use the float32 grain-cache precision before
float64 calculation. The output columns are ordered 0, 45 and 90 degrees.
`anchor_lower` and `anchor_upper` mean Sachs and Taylor, respectively; they are
reference estimates, not guaranteed bounds. Solver diagnostics are saved beside
the output, and failure to bracket a Taylor root stops the calculation.

For training on new data, insert these two arrays into the target bundle,
matching rows by `condition_ids`. For inference, use this anchor file directly.
The executable example performs both connections automatically and uses freshly
calculated references for all twenty conditions. Stored references serve only
as regression checks.

Train the three model instances by changing `--seed` to 7, 8, and 9. Each command runs one seed; the configuration list does not launch runs automatically:

```bash
python -m anchored_gnn.train \
  --graphs GRAPHS \
  --targets INPUT/targets.npz \
  --split configs/split.csv \
  --config configs/training.json \
  --seed 7 \
  --output RUNS/seed7
```

Inference needs graph files and an NPZ file containing only `condition_ids`,
`anchor_lower`, and `anchor_upper`:

```bash
python -m anchored_gnn.predict \
  --graphs GRAPHS \
  --anchors INPUT/anchors.npz \
  --checkpoint RUNS/seed7/model_best.pt \
  --output RUNS/seed7/predictions.npz
```

Evaluate predictions on the fixed test subset:

```bash
python -m anchored_gnn.evaluate \
  --predictions RUNS/seed7/predictions.npz \
  --targets INPUT/targets.npz \
  --split configs/split.csv \
  --group test
```

The complete small example can be rerun with its bundled files. It demonstrates
the software workflow and does not reproduce the manuscript's full-study results.
The full study data and checkpoints are not included.
For new data, the supplied graph builder requires grain geometry and
orientations; the supplied anchor module calculates the additional reference
estimates required for inference. CPFEM targets are needed for supervised
training and evaluation, but are not inputs to the anchor calculation or inference.

## Citation

Please cite the accompanying article and use the metadata in `CITATION.cff`.



## Figures for the executable example

The [four-panel figure](examples/expected/seed7/figures/example_panel_overview.png) includes hardening and Lankford parity for all three test textures and stress-locus evolution for two test textures. Individual 600-dpi PNG and vector SVG panels, separate legends, plotted data, and English/Korean captions are included under `examples/expected/seed7/figures/`.

Recreate the figures from the bundled results:

```bash
python examples/generate_figures.py
```

Recreate them from a fresh example training run:

```bash
python examples/generate_figures.py --results example_run --output new_figures
```

The supplied results in `examples/expected/seed7/` use seed 7 only. See the
[run record](examples/expected/seed7/RESULTS.md) for the completed and selected
epochs and the example metrics. No seed 8 or 9 results are bundled. Locus
visualization places predicted radii on the supplied CPFEM normal-stress
directions and displays Yld2000-2d fits. It does not independently predict the
stress directions. See the [figure caption](examples/expected/seed7/figures/CAPTIONS_EN.md).
