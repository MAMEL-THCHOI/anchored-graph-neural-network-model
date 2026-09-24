"""Compute rate-sensitive Sachs and full-constraint Taylor r-value references.

Uses grain orientations, volume fractions and the supplied 24 BCC slip systems.
No CPFEM responses, fitted r-values or trained neural-network weights are read.
The uniform-stress and uniform-strain references are not guaranteed r-value bounds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .graph import bunge_matrices, slip_families

ANGLES = (0.0, 45.0, 90.0)


def deviatoric_basis() -> np.ndarray:
    basis = np.zeros((5, 3, 3), dtype=np.float64)
    basis[0] = np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    basis[1] = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6.0)
    basis[2, 0, 1] = basis[2, 1, 0] = 1.0 / math.sqrt(2.0)
    basis[3, 0, 2] = basis[3, 2, 0] = 1.0 / math.sqrt(2.0)
    basis[4, 1, 2] = basis[4, 2, 1] = 1.0 / math.sqrt(2.0)
    return basis


DEV_BASIS = deviatoric_basis()


def slip_projection(eulers: np.ndarray, systems: np.ndarray) -> np.ndarray:
    families = slip_families(systems)
    normals = np.concatenate([families["110"][0], families["112"][0]])
    directions = np.concatenate([families["110"][1], families["112"][1]])
    rotations = bunge_matrices(np.asarray(eulers, dtype=np.float64))
    sample_n = np.einsum("gij,sj->gsi", rotations, normals)
    sample_d = np.einsum("gij,sj->gsi", rotations, directions)
    schmid = 0.5 * (
        np.einsum("gsi,gsj->gsij", sample_d, sample_n)
        + np.einsum("gsi,gsj->gsij", sample_n, sample_d)
    )
    return np.einsum("gsij,kij->gsk", schmid, DEV_BASIS)


def _constitutive(projection: np.ndarray, stress: np.ndarray, exponent: float) -> tuple[np.ndarray, np.ndarray]:
    resolved = np.einsum("gsk,gk->gs", projection, stress)
    absolute = np.maximum(np.abs(resolved), 1.0e-12)
    activity = resolved * absolute ** (exponent - 1.0)
    strain = np.einsum("gsk,gs->gk", projection, activity)
    tangent = exponent * np.einsum(
        "gsi,gsj,gs->gij", projection, projection, absolute ** (exponent - 1.0)
    )
    return strain, tangent


def invert_grain_stress(
    projection: np.ndarray,
    target: np.ndarray,
    exponent: float,
    max_iterations: int = 60,
    tolerance: float = 2.0e-8,
) -> tuple[np.ndarray, float]:
    """Invert the 24-slip power law for all grains in one vectorized solve."""

    target = np.asarray(target, dtype=np.float64)
    gram = np.einsum("gsi,gsj->gij", projection, projection)
    regularizer = 1.0e-10 * np.eye(5)[None, :, :]
    right_hand_side = np.broadcast_to(target, (len(projection), 5))[:, :, None]
    stress = np.linalg.solve(gram + regularizer, right_hand_side)[:, :, 0]
    schedule = [
        value
        for value in (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
        if value <= exponent
    ]
    if not schedule or schedule[-1] != float(exponent):
        schedule.append(float(exponent))
    target_norm = max(float(np.linalg.norm(target)), 1.0e-12)
    final_error = math.inf
    for current in schedule:
        strain, _ = _constitutive(projection, stress, current)
        scale = (target_norm / np.maximum(np.linalg.norm(strain, axis=1), 1.0e-12)) ** (1.0 / current)
        stress *= scale[:, None]
        for _ in range(max_iterations):
            strain, tangent = _constitutive(projection, stress, current)
            residual = strain - target[None, :]
            relative = np.linalg.norm(residual, axis=1) / target_norm
            final_error = float(np.max(relative))
            if final_error <= tolerance:
                break
            trace = np.trace(tangent, axis1=1, axis2=2)
            tangent = tangent + (1.0e-10 + 1.0e-8 * trace)[:, None, None] * np.eye(5)[None, :, :]
            step = np.linalg.solve(tangent, residual[:, :, None])[:, :, 0]
            step_norm = np.linalg.norm(step, axis=1)
            stress_norm = np.maximum(np.linalg.norm(stress, axis=1), 1.0e-8)
            alpha = np.minimum(1.0, 0.65 * stress_norm / np.maximum(step_norm, 1.0e-12))
            old_norm = np.linalg.norm(residual, axis=1)
            candidate = stress - alpha[:, None] * step
            for _ in range(10):
                candidate_strain, _ = _constitutive(projection, candidate, current)
                candidate_norm = np.linalg.norm(candidate_strain - target[None, :], axis=1)
                rejected = candidate_norm > old_norm * (1.0 - 1.0e-4 * alpha)
                if not bool(np.any(rejected)):
                    break
                alpha[rejected] *= 0.5
                candidate[rejected] = stress[rejected] - alpha[rejected, None] * step[rejected]
            stress = candidate
        if final_error > 2.0e-5:
            raise RuntimeError(f"Taylor grain inversion did not converge: n={current:g}, max_rel={final_error:.3e}")
    return stress, final_error


def strain_for_r(angle_deg: float, r_value: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    loading = np.asarray([math.cos(angle), math.sin(angle), 0.0])
    transverse = np.asarray([-math.sin(angle), math.cos(angle), 0.0])
    thickness = np.asarray([0.0, 0.0, 1.0])
    lateral = -float(r_value) / (1.0 + float(r_value))
    normal = -1.0 / (1.0 + float(r_value))
    strain = np.outer(loading, loading) + lateral * np.outer(transverse, transverse) + normal * np.outer(thickness, thickness)
    return np.einsum("ij,kij->k", strain, DEV_BASIS)


def taylor_r(
    eulers: np.ndarray,
    volumes: np.ndarray,
    exponent: float,
    systems: np.ndarray,
    angles: tuple[float, ...] = ANGLES,
) -> tuple[np.ndarray, dict]:
    projection = slip_projection(eulers, systems)
    weights = np.clip(np.asarray(volumes, dtype=np.float64), 0.0, None)
    weights /= max(float(weights.sum()), 1.0e-12)
    outputs = []
    diagnostics = {}
    for angle_deg in angles:
        angle = math.radians(float(angle_deg))
        transverse = np.asarray([-math.sin(angle), math.cos(angle), 0.0])
        thickness = np.asarray([0.0, 0.0, 1.0])
        cache: dict[float, tuple[float, float]] = {}

        def plane_stress_residual(log_r: float) -> float:
            key = round(float(log_r), 12)
            if key in cache:
                return cache[key][0]
            r_value = float(np.exp(log_r))
            target = strain_for_r(angle_deg, r_value)
            grain_stress, inversion_error = invert_grain_stress(projection, target, exponent)
            mean_coefficients = np.einsum("g,gk->k", weights, grain_stress)
            mean_stress = np.einsum("k,kij->ij", mean_coefficients, DEV_BASIS)
            residual = float(transverse @ mean_stress @ transverse - thickness @ mean_stress @ thickness)
            scale = max(float(np.linalg.norm(mean_stress)), 1.0e-12)
            cache[key] = (residual / scale, inversion_error)
            return cache[key][0]

        grid = np.log(np.asarray([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 100.0]))
        values = np.asarray([plane_stress_residual(value) for value in grid])
        brackets = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
        if len(brackets):
            index = int(brackets[np.argmin(np.abs(values[brackets]) + np.abs(values[brackets + 1]))])
            root = brentq(plane_stress_residual, float(grid[index]), float(grid[index + 1]), xtol=2.0e-5, maxiter=20)
            status = "bracketed_root"
        else:
            root = float(grid[int(np.argmin(np.abs(values)))])
            status = "grid_minimum_no_bracket"
        result = float(np.clip(np.exp(root), 0.02, 100.0))
        outputs.append(result)
        diagnostics[f"r{int(angle_deg):02d}"] = {
            "value": result,
            "status": status,
            "normalized_plane_stress_residual": float(plane_stress_residual(root)),
            "evaluations": len(cache),
            "max_grain_inversion_error": float(max(value[1] for value in cache.values())),
        }
    return np.asarray(outputs, dtype=np.float64), diagnostics


def sachs_r(eulers: np.ndarray, volumes: np.ndarray, exponent: float, systems: np.ndarray) -> np.ndarray:
    families = slip_families(systems)
    normals = np.concatenate([families["110"][0], families["112"][0]])
    directions = np.concatenate([families["110"][1], families["112"][1]])
    rotations = bunge_matrices(eulers)
    sample_n = np.einsum("gij,sj->gsi", rotations, normals)
    sample_d = np.einsum("gij,sj->gsi", rotations, directions)
    schmid_tensor = 0.5 * (
        np.einsum("gsi,gsj->gsij", sample_d, sample_n)
        + np.einsum("gsi,gsj->gsij", sample_n, sample_d)
    )
    weight = np.clip(np.asarray(volumes, dtype=np.float64), 0.0, None)
    weight /= max(float(weight.sum()), 1.0e-12)
    output = []
    for angle in ANGLES:
        radians = math.radians(angle)
        loading = np.asarray([math.cos(radians), math.sin(radians), 0.0])
        transverse = np.asarray([-math.sin(radians), math.cos(radians), 0.0])
        resolved = (
            np.einsum("gsi,i->gs", sample_n, loading)
            * np.einsum("gsi,i->gs", sample_d, loading)
        )
        activity = np.sign(resolved) * np.abs(resolved) ** exponent
        grain_rate = np.einsum("gs,gsij->gij", activity, schmid_tensor)
        mean_rate = np.einsum("g,gij->ij", weight, grain_rate)
        transverse_rate = float(transverse @ mean_rate @ transverse)
        thickness_rate = float(mean_rate[2, 2])
        output.append(float(np.clip(transverse_rate / thickness_rate, 0.02, 100.0)))
    return np.asarray(output, dtype=np.float64)


def references_from_microstructure(feature_ids, grain_eulers, systems):
    """Return Sachs, Taylor and solver diagnostics; Euler angles are in radians."""
    grid = np.asarray(feature_ids)
    eulers = np.asarray(grain_eulers)
    systems = np.asarray(systems, dtype=np.float64)
    if grid.ndim != 3 or not np.issubdtype(grid.dtype, np.integer) or grid.size == 0:
        raise ValueError('feature_ids must be a nonempty 3D integer grain map.')
    if np.any(grid <= 0):
        raise ValueError('Each voxel must belong to a positive grain ID.')
    ids, counts = np.unique(grid, return_counts=True)
    if eulers.ndim != 2 or eulers.shape[1] != 3 or ids[-1] >= len(eulers):
        raise ValueError('grain_eulers must have three columns and a row for every grain ID.')
    if not np.isfinite(eulers[ids]).all():
        raise ValueError('Grain orientations must be finite.')
    if (systems.shape != (24, 6) or not np.isfinite(systems).all()
            or np.any(np.linalg.norm(systems[:, :3], axis=1) == 0)
            or np.any(np.linalg.norm(systems[:, 3:], axis=1) == 0)):
        raise ValueError('Supply 24 finite, nonzero BCC slip normals and directions.')
    # Match the float32 orientation and volume convention of the grain cache.
    orientations = eulers[ids].astype(np.float32).astype(np.float64)
    volumes = (counts / grid.size).astype(np.float32).astype(np.float64)
    lower = sachs_r(orientations, volumes, 20.0, systems)
    upper, diagnostics = taylor_r(orientations, volumes, 8.0, systems)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError('Non-finite anchor estimate.')
    if any(item['status'] != 'bracketed_root' for item in diagnostics.values()):
        raise RuntimeError(f'Taylor plane-stress root was not bracketed: {diagnostics}')
    return lower, upper, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--microstructures', type=Path, required=True)
    parser.add_argument('--slip-systems', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('--output must end in .npz')
    paths = sorted(args.microstructures.glob('*.npz'))
    if not paths:
        raise FileNotFoundError('No microstructure NPZ files found.')
    diagnostic_path = args.output.with_suffix('.diagnostics.json')
    if args.output.exists() or diagnostic_path.exists():
        raise FileExistsError('Choose a new output path.')
    systems = np.loadtxt(args.slip_systems)
    lower, upper, diagnostics = [], [], {}
    for path in paths:
        print(f'Calculating Sachs/Taylor references: {path.stem}', flush=True)
        with np.load(path, allow_pickle=False) as raw:
            sachs, taylor, info = references_from_microstructure(
                raw['feature_ids'], raw['grain_eulers'], systems)
        lower.append(sachs)
        upper.append(taylor)
        diagnostics[path.stem] = info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, condition_ids=[p.stem for p in paths],
                        anchor_lower=lower, anchor_upper=upper)
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

