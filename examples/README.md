# Complete small-data example

This example calculates Sachs/Taylor references and rebuilds grain graphs from twenty microstructures, trains a fresh anchored grain graph model on fourteen textures, selects a checkpoint on three validation textures, and predicts three held-out test textures. All required microstructure and response arrays are included. No solver or private project files are read by the example.

## Run from the repository root

```bash
python -m pip install -r examples/requirements.txt
python examples/run_example.py --seed 7
```

The second command creates `example_run/`, including freshly calculated anchors, training results and parity/locus figures under `example_run/figures/`. An existing output directory is never overwritten. To repeat the experiment, choose a new directory:

```bash
python examples/run_example.py --seed 7 --output another_example_run
```

The script calls the public anchor calculation, graph builder, training, prediction, and evaluation modules. Normalization statistics are fitted on the training subset. Runs default to seed 7, and the configuration lists the manuscript seeds 7, 8 and 9 for independent runs. The commands above explicitly use seed 7 to reproduce the bundled reference. Each invocation runs one seed using four numerical-library threads; it does not load the paper's trained weights. The saved checkpoint contains statistics fitted only on the fourteen training textures. Validation selects the checkpoint. Test responses are used for the final comparison only.

## Fixed example conditions

| Subset | Conditions |
|---|---|
| Training | TS002, TS023, TS049, TS071, TS097, TS115, TS138, TS157, TS177, TS198, TS221, TS244, TS271, TS297 |
| Validation | TS001, TS167, TS300 |
| Test | TS003, TS151, TS298 |

The original split memberships were retained. Within each subset, condition IDs were sorted and evenly spaced entries were chosen. The selection was fixed before inspecting response values or prediction errors. The exact rule is in `selection.json`. Each condition uses its first RVE realization.

## Supplied files

| File or directory | Contents |
|---|---|
| `microstructures/` | Twenty compressed NumPy files: `feature_ids`, the integer grain-ID grid in z/y/x order, and `grain_eulers`, a grain-ID-indexed table of Bunge angles in radians. Row zero is unused. |
| `grain_orientations.csv` | Readable grain orientations, volume fractions and periodic centroids in normalized x/y/z coordinates. |
| `slip_systems.txt` | Twenty-four rows of slip-plane normal and slip-direction components, in that order. |
| `graphs/` | Graphs constructed from those microstructures. The example rebuilds and checks these arrays before training. |
| `targets.npz` | All postprocessed values required by the existing joint loss: hardening parameters, stress radii and corresponding strain coordinates, Lankford coefficients, two anchors and auxiliary responses. |
| `cpfem_hardening.csv` | Readable CPFEM stress radii, strain coordinates and stress components at each supplied plastic-work level and loading angle. |
| `cpfem_r_and_anchors.csv` | Readable CPFEM r-values and Sachs/Taylor reference estimates. |
| `test_anchors.npz` | Only the three test condition IDs and their anchor estimates, for inference. It contains no CPFEM r-value targets. |
| `split.csv` | The twenty-condition example split, used instead of the full-study split under `configs/`. |
| `expected/seed7/` | The independently executed example's checkpoint, predictions, metrics, training history, comparison tables and plot. |

For targets, `hardening_parameters[:, :, angle]` contains initial stress, stress increment, rate parameter and exponent. The loading-angle order is 0, 20, ..., 160 degrees; work levels are 0.25, 0.5, 0.75, 1, 2, 5 and 10 MJ/m³. Lankford directions are 0, 45 and 90 degrees. `stress_aux` concatenates the flattened stress-angle coordinate atan2(sigma22,sigma11) and shear stress divided by sqrt(sigma11² + sigma22² + 2 sigma12²). `flow_aux` contains tensile flow stresses at axial logarithmic strains 0.02, 0.05 and 0.08, ordered first by tensile direction and then strain. These are fixed postprocessed input values, not values fitted using the test predictions.

The decoder input names `anchor_lower` and `anchor_upper` correspond to Sachs and Taylor estimates, respectively. They are reference responses; these names do not imply proven mathematical bounds. The example recalculates both from grain orientations, volume fractions and the supplied slip systems. Stored values are used only to check agreement. Newly calculated values are written to `anchors.npz`, inserted into the run's `targets.npz` for training, and selected into `test_anchors.npz` for inference. `anchor_diagnostics.json` records agreement and Taylor convergence. No CPFEM response is used to calculate these references. The existing postprocessed CPFEM targets still provide supervision and evaluation; this example does not run CPFEM itself.

For a separate calculation on new microstructures, use `python -m anchored_gnn.anchors --microstructures INPUT/microstructures --slip-systems examples/slip_systems.txt --output anchors.npz`. Details of the reference model and precision convention are in the repository README.

## Expected output

Open `expected/seed7/RESULTS.md` or `expected/seed7/example_results.png` to inspect the bundled run. A fresh run produces the same file types in the chosen output directory. The demo checkpoints and results come from this small dataset, not the paper's full training campaign.

This example demonstrates the complete software workflow. Its three test RVEs are not sufficient to establish generalization performance, and its scores are not the paper's reported full-database scores.


## Additional parity and locus figures

`expected/seed7/figures/example_panel_overview.png` is the complete attachable figure. Individual panels, SVG files, plotted data and captions are stored alongside it. From the repository root, run `python examples/generate_figures.py` to reproduce the bundled figure in a new `figure_run/` directory. To plot a fresh training run, use `python examples/generate_figures.py --results example_run --output new_figures`.

The supplied example results use only seed 7. The completed and selected epochs
are recorded in [the run report](expected/seed7/RESULTS.md). They are a single run,
not an ensemble. To start a new run with the default seed 7, omit `--seed`; use
`--seed 8` or `--seed 9` and a new output directory for the other configured seeds.
Reports and figure captions read the actual seed and selected epoch from each
run. No results for seeds 8 or 9 are bundled.

The versions used for the bundled run are listed in `tested_environment.json`.
Floating-point and optimization trajectories can vary with numerical-library
versions and hardware; a fixed seed does not guarantee bitwise agreement across
all environments.
