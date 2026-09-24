"""Minimal implementation check; no research data are loaded."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchored_gnn.graph import bunge_matrices, cubic_misorientation, face_pairs
from anchored_gnn.model import AnchoredGrainGraph
from anchored_gnn.workflow import metric
from anchored_gnn.train import validation_score


torch.manual_seed(4)
node_count = 6
batch = {
    "x": torch.randn(node_count, 220),
    "edge_index": torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
    "edge_attr": torch.randn(6, 6),
    "volume": torch.full((node_count,), 1.0 / 3.0),
    "graph_id": torch.tensor([0, 0, 0, 1, 1, 1]),
    "graph_count": 2,
}
model = AnchoredGrainGraph(lambda_init=0.4)
assert sum(p.numel() for p in model.parameters()) == 51623
output = model(batch, torch.zeros(2, 3), torch.ones(2, 3))
assert output["coefficients"].shape == (2, 36)
assert output["log_r"].shape == (2, 3)
loss = output["log_r"].square().mean() + output["coefficients"].square().mean()
loss.backward()
assert all(torch.isfinite(value).all() for value in output.values())
assert metric(np.arange(5.0), np.arange(5.0))["R2"] == 1.0
# Checkpoint selection follows relative errors, not absolute-error units.
first = {"hardening": {"mre": 0.01, "mae": 10.0}, "r": {"mre": 0.03, "mae": 0.2}}
second = {"hardening": {"mre": 0.02, "mae": 1.0}, "r": {"mre": 0.03, "mae": 0.1}}
assert validation_score(first) < validation_score(second)
assert abs(validation_score(first) - 4.0) < 1e-12
pairs, counts = face_pairs(np.asarray([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]))
assert pairs.shape == (1, 2) and int(counts[0]) > 0

# Equivalent cubic orientations must describe the same physical orientation.
g, other, q = bunge_matrices(np.deg2rad([[31, 47, 13], [73, 28, 19], [12, 43, 61]]))
cubic = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
rotations = np.stack([g, g @ cubic, other])
source, target = np.asarray([0, 0]), np.asarray([1, 2])
angles = cubic_misorientation(rotations, source, target)
np.testing.assert_allclose(angles[0], 0., atol=2e-8)
# A common specimen rotation must not change the disorientation.
np.testing.assert_allclose(cubic_misorientation(q @ rotations, source, target), angles, atol=2e-8)
np.testing.assert_allclose(cubic_misorientation(rotations, target, source), angles, atol=2e-8)
from anchored_gnn.anchors import references_from_microstructure, sachs_r, taylor_r

# A single Cube orientation has equal width/thickness contraction in RD and TD.
systems = np.loadtxt(Path(__file__).resolve().parents[1] / 'examples/slip_systems.txt')
cube = np.zeros((2, 3))
sachs = sachs_r(cube[1:], np.ones(1), 20., systems)
taylor, diagnostics = taylor_r(cube[1:], np.ones(1), 8., systems, angles=(0., 90.))
np.testing.assert_allclose(sachs[[0, 2]], 1., atol=2e-8)
np.testing.assert_allclose(taylor, 1., atol=2e-5)
assert all(item['status'] == 'bracketed_root' for item in diagnostics.values())
try:
    references_from_microstructure(np.zeros((1, 1, 1), dtype=int), cube, systems)
except ValueError:
    pass
else:
    raise AssertionError('An unassigned voxel must be rejected.')
print("smoke test passed")
