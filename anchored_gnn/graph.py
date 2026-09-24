"""Construct the grain graph used by the anchored surrogate.

The builder accepts NumPy arrays rather than solver files. Grain nodes contain
volume fraction, periodic centroid coordinates, and rank-ordered Schmid
descriptors. Directed edges represent shared voxel faces, including contacts
across opposite faces of the periodic cell.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np


ANGLES_DEG = tuple(float(value) for value in range(0, 180, 20))


def bunge_matrices(eulers: np.ndarray) -> np.ndarray:
    """Map crystal vectors to the specimen frame from Bunge angles in radians."""
    eulers = np.asarray(eulers, dtype=np.float64)
    if eulers.ndim != 2 or eulers.shape[1] != 3:
        raise ValueError(f"eulers must have shape (n, 3), found {eulers.shape}")
    matrices = np.empty((len(eulers), 3, 3), dtype=np.float64)
    for index, (phi1, phi, phi2) in enumerate(eulers):
        c1, s1 = math.cos(phi1), math.sin(phi1)
        c, s = math.cos(phi), math.sin(phi)
        c2, s2 = math.cos(phi2), math.sin(phi2)
        rz1 = np.asarray([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
        rx = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        rz2 = np.asarray([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
        matrices[index] = rz1 @ rx @ rz2
    return matrices


def proper_cubic_rotations() -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    eye = np.eye(3)
    for permutation in itertools.permutations(range(3)):
        base = eye[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    return rotations


def slip_families(systems: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Validate and separate 24 BCC slip systems into two 12-system families."""
    systems = np.asarray(systems, dtype=np.float64)
    if systems.shape != (24, 6):
        raise ValueError(f"slip_systems must have shape (24, 6), found {systems.shape}")
    raw_normals, raw_directions = systems[:, :3], systems[:, 3:]
    normals = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]
    directions = raw_directions / np.linalg.norm(raw_directions, axis=1)[:, None]
    if np.max(np.abs(np.einsum("ij,ij->i", normals, directions))) > 1.0e-10:
        raise ValueError("slip plane normals and directions must be orthogonal")
    signatures = [tuple(sorted(int(round(abs(value))) for value in row)) for row in raw_normals]
    output = {}
    for name, signature in (("110", (0, 1, 1)), ("112", (1, 1, 2))):
        mask = np.asarray([value == signature for value in signatures])
        if int(mask.sum()) != 12:
            raise ValueError(f"expected 12 systems in the {name} family")
        output[name] = (normals[mask], directions[mask])
    return output


def schmid_descriptors(rotations: np.ndarray, systems: np.ndarray) -> np.ndarray:
    """Return 216 family-separated and rank-ordered Schmid channels per grain."""
    families = slip_families(systems)
    blocks = []
    for family in ("110", "112"):
        normals, directions = families[family]
        sample_normals = np.einsum("nij,sj->nsi", rotations, normals)
        sample_directions = np.einsum("nij,sj->nsi", rotations, directions)
        family_blocks = []
        for angle in ANGLES_DEG:
            theta = math.radians(angle)
            load = np.asarray([math.cos(theta), math.sin(theta), 0.0])
            factors = np.abs(
                np.einsum("nsi,i->ns", sample_normals, load)
                * np.einsum("nsi,i->ns", sample_directions, load)
            )
            family_blocks.append(np.sort(factors, axis=1)[:, ::-1])
        blocks.append(np.concatenate(family_blocks, axis=1))
    return np.concatenate(blocks, axis=1)


def periodic_centroids(feature_ids: np.ndarray, grain_ids: np.ndarray) -> np.ndarray:
    shape = np.asarray(feature_ids.shape, dtype=float)
    indices = np.indices(feature_ids.shape, dtype=float)
    centroids = np.zeros((len(grain_ids), 3), dtype=np.float64)
    for row, grain_id in enumerate(grain_ids):
        mask = feature_ids == grain_id
        for axis in range(3):
            coordinate = (indices[axis][mask] + 0.5) / shape[axis]
            angle = 2.0 * math.pi * coordinate
            mean_sin, mean_cos = np.mean(np.sin(angle)), np.mean(np.cos(angle))
            if math.hypot(mean_sin, mean_cos) < 1.0e-8:
                centroids[row, axis] = float(np.mean(coordinate)) % 1.0
            else:
                centroids[row, axis] = math.atan2(mean_sin, mean_cos) / (2.0 * math.pi) % 1.0
    return centroids[:, ::-1]


def face_pairs(feature_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for axis in range(3):
        pairs.append(
            np.column_stack(
                [
                    np.take(feature_ids, np.arange(feature_ids.shape[axis] - 1), axis=axis).ravel(),
                    np.take(feature_ids, np.arange(1, feature_ids.shape[axis]), axis=axis).ravel(),
                ]
            )
        )
        pairs.append(
            np.column_stack(
                [np.take(feature_ids, [-1], axis=axis).ravel(), np.take(feature_ids, [0], axis=axis).ravel()]
            )
        )
    pairs = np.concatenate(pairs).astype(np.int64)
    keep = (pairs[:, 0] > 0) & (pairs[:, 1] > 0) & (pairs[:, 0] != pairs[:, 1])
    return np.unique(np.sort(pairs[keep], axis=1), axis=0, return_counts=True)


def cubic_misorientation(rotations: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return cubic disorientation angles divided by pi.

    Crystal symmetry acts in the crystal frame, so use g_source.T @ g_target
    for the crystal-to-specimen rotation convention above.
    """
    delta = np.einsum("eji,ejk->eik", rotations[source], rotations[target], optimize=True)
    best_cosine = np.full(len(source), -1.0)
    for symmetry in proper_cubic_rotations():
        trace = np.einsum("ab,eba->e", symmetry, delta, optimize=True)
        best_cosine = np.maximum(best_cosine, 0.5 * (trace - 1.0))
    return np.arccos(np.clip(best_cosine, -1.0, 1.0)) / math.pi


def build_grain_graph(
    feature_ids: np.ndarray, grain_eulers: np.ndarray, slip_systems: np.ndarray
) -> dict[str, np.ndarray]:
    """Build one periodic graph from a voxel grain map and grain orientations.

    ``grain_eulers`` is indexed by the positive integer IDs in ``feature_ids``.
    Euler angles must be in radians. The returned node and edge widths are 220
    and 6, respectively.
    """
    feature_ids = np.asarray(feature_ids, dtype=np.int64)
    if feature_ids.ndim != 3:
        raise ValueError("feature_ids must be a three-dimensional array")
    grain_ids = np.asarray(sorted(int(value) for value in np.unique(feature_ids) if value > 0))
    if not len(grain_ids) or grain_ids.max() >= len(grain_eulers):
        raise ValueError("grain_eulers is not indexed for all positive feature IDs")
    eulers = np.asarray(grain_eulers, dtype=np.float64)[grain_ids]
    rotations = bunge_matrices(eulers)
    schmid = schmid_descriptors(rotations, slip_systems)
    centroids = periodic_centroids(feature_ids, grain_ids)
    volume = np.asarray([(feature_ids == value).sum() for value in grain_ids], dtype=np.float64)
    volume /= float(feature_ids.size)

    grain_pairs, shared = face_pairs(feature_ids)
    lookup = np.full(int(grain_ids.max()) + 1, -1, dtype=np.int64)
    lookup[grain_ids] = np.arange(len(grain_ids))
    undirected = lookup[grain_pairs]
    source = np.concatenate([undirected[:, 0], undirected[:, 1]])
    target = np.concatenate([undirected[:, 1], undirected[:, 0]])
    shared = np.concatenate([shared, shared]).astype(np.float64)
    delta = centroids[target] - centroids[source]
    delta -= np.round(delta)
    misorientation = cubic_misorientation(rotations, source, target)
    schmid_delta = np.linalg.norm(schmid[target] - schmid[source], axis=1) / math.sqrt(schmid.shape[1])

    node_features = np.column_stack([volume, centroids, schmid]).astype(np.float32)
    edge_features = np.column_stack(
        [shared / max(float(shared.max()), 1.0), misorientation, delta, schmid_delta]
    ).astype(np.float32)
    return {
        "x": node_features,
        "edge_index": np.vstack([source, target]).astype(np.int64),
        "edge_attr": edge_features,
        "volume": volume.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ids", type=Path, required=True, help="3D integer .npy array")
    parser.add_argument("--eulers", type=Path, required=True, help="grain-indexed Bunge Euler .npy array in radians")
    parser.add_argument("--slip-systems", type=Path, required=True, help="24 by 6 text array")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = build_grain_graph(np.load(args.feature_ids), np.load(args.eulers), np.loadtxt(args.slip_systems))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **graph)


if __name__ == "__main__":
    main()
