"""Data, normalization, physics, loss, and metric utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gamma, gammainc
from torch.utils.data import DataLoader, Dataset


HS_BOUNDS = np.asarray(
    [(1.0, 1000.0), (1.0e-6, 3000.0), (1.0e-3, 1.0e5), (0.1, 3.0)], dtype=np.float64
)
REQUIRED_TARGETS = {
    "condition_ids",
    "hardening_parameters",
    "hardening_radius",
    "hardening_strain",
    "r_values",
    "anchor_lower",
    "anchor_upper",
    "stress_aux",
    "flow_aux",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"condition_id", "split"}:
        raise ValueError("split CSV must contain only condition_id and split columns")
    result = {row["condition_id"]: row["split"] for row in rows}
    if len(result) != len(rows) or set(result.values()) != {"train", "val", "test"}:
        raise ValueError("split CSV contains duplicate IDs or invalid split labels")
    return result


def load_targets(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = REQUIRED_TARGETS - set(source.files)
        if missing:
            raise ValueError(f"target bundle is missing: {sorted(missing)}")
        targets = {key: np.asarray(source[key]) for key in REQUIRED_TARGETS}
    targets["condition_ids"] = np.asarray([str(value) for value in targets["condition_ids"]])
    count = len(targets["condition_ids"])
    expected = {
        "hardening_parameters": (count, 4, 9),
        "hardening_radius": (count, 7, 9),
        "hardening_strain": (count, 7, 9),
        "r_values": (count, 3),
        "anchor_lower": (count, 3),
        "anchor_upper": (count, 3),
        "stress_aux": (count, 126),
        "flow_aux": (count, 9),
    }
    for key, shape in expected.items():
        targets[key] = np.asarray(targets[key], dtype=np.float64)
        if targets[key].shape != shape or not np.isfinite(targets[key]).all():
            raise ValueError(f"{key} must be a finite array with shape {shape}")
    return targets


@dataclass
class GraphItem:
    condition_id: str
    target_index: int
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    volume: torch.Tensor


class GraphDataset(Dataset):
    def __init__(self, items: list[GraphItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> GraphItem:
        return self.items[index]


def collate(items: list[GraphItem]) -> dict[str, torch.Tensor | list[str] | int]:
    xs, edges, attrs, volumes, graph_ids, indices, ids = [], [], [], [], [], [], []
    offset = 0
    for graph_index, item in enumerate(items):
        node_count = len(item.x)
        xs.append(item.x)
        edges.append(item.edge_index + offset)
        attrs.append(item.edge_attr)
        volumes.append(item.volume)
        graph_ids.append(torch.full((node_count,), graph_index, dtype=torch.long))
        indices.append(item.target_index)
        ids.append(item.condition_id)
        offset += node_count
    return {
        "x": torch.cat(xs),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(attrs),
        "volume": torch.cat(volumes),
        "graph_id": torch.cat(graph_ids),
        "target_index": torch.tensor(indices, dtype=torch.long),
        "condition_ids": ids,
        "graph_count": len(items),
    }


def load_raw_graphs(graph_dir: Path, condition_ids: list[str]) -> list[dict[str, np.ndarray | str]]:
    graphs = []
    for condition_id in condition_ids:
        path = graph_dir / f"{condition_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {"x", "edge_index", "edge_attr", "volume"}
            if required - set(source.files):
                raise ValueError(f"{path} does not follow the graph schema")
            graph = {key: np.asarray(source[key]) for key in required}
        if graph["x"].ndim != 2 or graph["x"].shape[1] != 220:
            raise ValueError(f"{path}: x must have shape (nodes, 220)")
        if graph["edge_attr"].ndim != 2 or graph["edge_attr"].shape[1] != 6:
            raise ValueError(f"{path}: edge_attr must have shape (edges, 6)")
        if graph["edge_index"].shape != (2, len(graph["edge_attr"])):
            raise ValueError(f"{path}: edge_index must have shape (2, edges)")
        if graph["volume"].shape != (len(graph["x"]),):
            raise ValueError(f"{path}: volume must have shape (nodes,)")
        graphs.append({"condition_id": condition_id, **graph})
    return graphs


def feature_stats(graphs: list[dict], train_ids: set[str]) -> dict[str, np.ndarray]:
    train = [graph for graph in graphs if graph["condition_id"] in train_ids]
    if not train:
        raise ValueError("no training graphs were found")
    nodes = np.concatenate([graph["x"] for graph in train]).astype(np.float64)
    edges = np.concatenate([graph["edge_attr"] for graph in train]).astype(np.float64)
    return {
        "x_mean": nodes.mean(axis=0).astype(np.float32),
        "x_std": np.maximum(nodes.std(axis=0), 1.0e-6).astype(np.float32),
        "edge_mean": edges.mean(axis=0).astype(np.float32),
        "edge_std": np.maximum(edges.std(axis=0), 1.0e-6).astype(np.float32),
    }


def graph_items(
    graphs: list[dict], target_index: dict[str, int], stats: dict[str, np.ndarray]
) -> list[GraphItem]:
    items = []
    for graph in graphs:
        node = np.clip((graph["x"] - stats["x_mean"]) / stats["x_std"], -8.0, 8.0)
        edge = np.clip((graph["edge_attr"] - stats["edge_mean"]) / stats["edge_std"], -8.0, 8.0)
        items.append(
            GraphItem(
                condition_id=graph["condition_id"],
                target_index=target_index[graph["condition_id"]],
                x=torch.tensor(node, dtype=torch.float32),
                edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(edge, dtype=torch.float32),
                volume=torch.tensor(graph["volume"], dtype=torch.float32),
            )
        )
    return items


def make_loaders(
    items: list[GraphItem], split_map: dict[str, str], batch_size: int, seed: int
) -> dict[str, DataLoader]:
    grouped = {name: [item for item in items if split_map[item.condition_id] == name] for name in split_map.values()}
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(
            GraphDataset(grouped[name]),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            collate_fn=collate,
            num_workers=0,
        )
        for name in ("train", "val", "test")
    }


def fourier_basis(order: int = 4) -> torch.Tensor:
    radians = np.radians(np.arange(0.0, 180.0, 20.0))
    columns = [np.ones_like(radians)]
    for harmonic in range(1, order + 1):
        columns.extend([np.cos(2 * harmonic * radians), np.sin(2 * harmonic * radians)])
    return torch.tensor(np.column_stack(columns), dtype=torch.float32)


def fit_target_stats(targets: dict[str, np.ndarray], train_indices: np.ndarray) -> dict[str, np.ndarray | float]:
    parameters = np.log(np.clip(targets["hardening_parameters"], HS_BOUNDS[:, 0, None], HS_BOUNDS[:, 1, None]))
    parameter_mean = parameters[train_indices].mean(axis=(0, 2))
    parameter_std = np.maximum(parameters[train_indices].std(axis=(0, 2)), 1.0e-6)
    log_radius = np.log(np.maximum(targets["hardening_radius"], 1.0e-8))
    radius_mean = log_radius[train_indices].mean(axis=(0, 2))
    radius_std = np.maximum(log_radius[train_indices].std(axis=(0, 2)), 1.0e-6)
    stress_mean = targets["stress_aux"][train_indices].mean(axis=0)
    stress_std = np.maximum(targets["stress_aux"][train_indices].std(axis=0), 1.0e-5)
    flow_mean = targets["flow_aux"][train_indices].mean(axis=0)
    flow_std = np.maximum(targets["flow_aux"][train_indices].std(axis=0), 1.0e-5)
    log_r = np.log(np.clip(targets["r_values"][train_indices], 0.02, 100.0))
    return {
        "parameter_mean": parameter_mean,
        "parameter_std": parameter_std,
        "radius_mean": radius_mean,
        "radius_std": radius_std,
        "stress_mean": stress_mean,
        "stress_std": stress_std,
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "r_scale": np.maximum(log_r.std(axis=0), 0.05),
        "lambda_init": 0.5,
    }


def normalized_target_tensors(targets: dict[str, np.ndarray], stats: dict) -> dict[str, torch.Tensor]:
    parameters = np.log(np.clip(targets["hardening_parameters"], HS_BOUNDS[:, 0, None], HS_BOUNDS[:, 1, None]))
    log_radius = np.log(np.maximum(targets["hardening_radius"], 1.0e-8))
    arrays = {
        "parameter": (parameters - stats["parameter_mean"][None, :, None]) / stats["parameter_std"][None, :, None],
        "radius": (log_radius - stats["radius_mean"][None, :, None]) / stats["radius_std"][None, :, None],
        "strain": targets["hardening_strain"],
        "r": targets["r_values"],
        "anchor_lower": targets["anchor_lower"],
        "anchor_upper": targets["anchor_upper"],
        "stress_aux": (targets["stress_aux"] - stats["stress_mean"]) / stats["stress_std"],
        "flow_aux": (targets["flow_aux"] - stats["flow_mean"]) / stats["flow_std"],
    }
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in arrays.items()}


def inverse_hs_torch(transformed: torch.Tensor) -> torch.Tensor:
    bounds = torch.tensor(HS_BOUNDS, dtype=transformed.dtype, device=transformed.device)
    values = torch.exp(torch.clamp(transformed, -16.0, 16.0))
    return torch.maximum(torch.minimum(values, bounds[None, :, 1, None]), bounds[None, :, 0, None])


def hs_radius_torch(parameters: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
    sigma_y, delta, rate, exponent = [parameters[:, index, None, :] for index in range(4)]
    strain = torch.clamp(strain, min=0.0)
    return sigma_y + delta * (1.0 - torch.exp(-rate * strain**exponent))


def hs_work_numpy(parameters: np.ndarray, strain: np.ndarray) -> np.ndarray:
    sigma_y, delta, rate, exponent = [parameters[:, index, None, :] for index in range(4)]
    strain = np.maximum(strain, 0.0)
    shape = 1.0 / exponent
    integral = gamma(shape) * gammainc(shape, rate * strain**exponent) / (exponent * rate**shape)
    return (sigma_y + delta) * strain - delta * integral


def hs_radius_numpy(parameters: np.ndarray, strain: np.ndarray) -> np.ndarray:
    sigma_y, delta, rate, exponent = [parameters[:, index, None, :] for index in range(4)]
    return sigma_y + delta * (1.0 - np.exp(-rate * np.maximum(strain, 0.0) ** exponent))


def radius_at_work(parameters: np.ndarray, work_levels: np.ndarray) -> np.ndarray:
    target = np.asarray(work_levels, dtype=np.float64)[None, :, None]
    shape = (len(parameters), len(work_levels), parameters.shape[2])
    low, high = np.zeros(shape), np.full(shape, 0.08)
    for _ in range(12):
        insufficient = hs_work_numpy(parameters, high) < target
        high = np.where(insufficient, 2.0 * high, high)
    for _ in range(48):
        middle = 0.5 * (low + high)
        below = hs_work_numpy(parameters, middle) < target
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    return hs_radius_numpy(parameters, 0.5 * (low + high))


def decode_parameters(coefficients: torch.Tensor, basis: torch.Tensor, stats: dict) -> torch.Tensor:
    normalized = torch.einsum("gpb,ab->gpa", coefficients.reshape(len(coefficients), 4, basis.shape[1]), basis)
    mean = torch.as_tensor(stats["parameter_mean"], dtype=normalized.dtype, device=normalized.device)
    std = torch.as_tensor(stats["parameter_std"], dtype=normalized.dtype, device=normalized.device)
    return inverse_hs_torch(normalized * std[None, :, None] + mean[None, :, None])


def joint_loss(
    output: dict[str, torch.Tensor],
    indices: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    stats: dict,
    basis: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    normalized_parameters = torch.einsum(
        "gpb,ab->gpa", output["coefficients"].reshape(len(indices), 4, basis.shape[1]), basis
    )
    device = output["coefficients"].device
    mean = torch.as_tensor(stats["parameter_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(stats["parameter_std"], dtype=torch.float32, device=device)
    parameters = inverse_hs_torch(normalized_parameters * std[None, :, None] + mean[None, :, None])
    radius = hs_radius_torch(parameters, tensors["strain"][indices]).clamp(1.0e-6, 5000.0)
    radius_mean = torch.as_tensor(stats["radius_mean"], dtype=torch.float32, device=device)
    radius_std = torch.as_tensor(stats["radius_std"], dtype=torch.float32, device=device)
    normalized_radius = (torch.log(radius.clamp_min(1.0e-6)) - radius_mean[None, :, None]) / radius_std[None, :, None]
    parameter_loss = F.smooth_l1_loss(normalized_parameters, tensors["parameter"][indices], beta=1.0)
    curve_loss = F.smooth_l1_loss(normalized_radius, tensors["radius"][indices], beta=1.0)
    hardening_loss = curve_loss + weights["parameter"] * parameter_loss
    log_actual = torch.log(tensors["r"][indices].clamp(0.02, 100.0))
    r_scale = torch.as_tensor(stats["r_scale"], dtype=torch.float32, device=device)
    r_loss = F.smooth_l1_loss((output["log_r"] - log_actual) / r_scale[None, :], torch.zeros_like(log_actual), beta=0.5)
    stress_loss = F.smooth_l1_loss(output["stress_aux"], tensors["stress_aux"][indices], beta=1.0)
    flow_loss = F.smooth_l1_loss(output["flow_aux"], tensors["flow_aux"][indices], beta=1.0)
    auxiliary_loss = 0.5 * (stress_loss + flow_loss)
    total = (
        weights["hardening"] * hardening_loss
        + weights["r"] * r_loss
        + weights["auxiliary"] * auxiliary_loss
    )
    parts = {"hardening": float(hardening_loss.detach()), "r": float(r_loss.detach()), "auxiliary": float(auxiliary_loss.detach())}
    return total, parts


def metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual, predicted = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    residual = predicted - actual
    denominator = max(float(np.sum((actual - actual.mean()) ** 2)), 1.0e-12)
    return {
        "count": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "mre": float(np.mean(np.abs(residual) / np.maximum(np.abs(actual), 1.0e-12))),
        "R2": float(1.0 - np.sum(residual**2) / denominator),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
