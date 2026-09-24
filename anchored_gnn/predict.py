"""Run inference with a trained anchored grain graph model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import AnchoredGrainGraph
from .workflow import (
    GraphDataset,
    collate,
    decode_parameters,
    fourier_basis,
    graph_items,
    load_raw_graphs,
    move_batch,
    radius_at_work,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True, help="NPZ with condition_ids and two anchor arrays")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, target_stats = checkpoint["config"], checkpoint["target_stats"]
    with np.load(args.anchors, allow_pickle=False) as source:
        required = {"condition_ids", "anchor_lower", "anchor_upper"}
        if required - set(source.files):
            raise ValueError("anchor bundle must contain condition_ids, anchor_lower, and anchor_upper")
        condition_ids = [str(value) for value in source["condition_ids"]]
        lower = np.asarray(source["anchor_lower"], dtype=np.float32)
        upper = np.asarray(source["anchor_upper"], dtype=np.float32)
    if lower.shape != (len(condition_ids), 3) or upper.shape != lower.shape:
        raise ValueError("anchor arrays must have shape (conditions, 3)")
    index = {value: row for row, value in enumerate(condition_ids)}
    graphs = load_raw_graphs(args.graphs, condition_ids)
    items = graph_items(graphs, index, checkpoint["feature_stats"])
    loader = DataLoader(GraphDataset(items), batch_size=int(config["batch_size"]), collate_fn=collate)
    basis = fourier_basis(int(config["fourier_order"])).to(device)
    model = AnchoredGrainGraph(
        coefficient_count=4 * basis.shape[1],
        lambda_init=float(target_stats["lambda_init"]),
        hidden=int(config["hidden"]),
        dropout=float(config["dropout"]),
        message_layers=int(config["message_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    parameter_rows, hardening_rows, r_rows, mixing_rows, correction_rows = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            indices = batch["target_index"].cpu().numpy()
            log_lower = torch.log(torch.tensor(lower[indices], device=device).clamp(0.02, 100.0))
            log_upper = torch.log(torch.tensor(upper[indices], device=device).clamp(0.02, 100.0))
            output = model(batch, log_lower, log_upper)
            parameters = decode_parameters(output["coefficients"], basis, target_stats).cpu().numpy()
            parameter_rows.append(parameters)
            hardening_rows.append(radius_at_work(parameters, np.asarray(config["work_levels"])))
            r_rows.append(torch.exp(torch.clamp(output["log_r"], -4.0, 5.0)).cpu().numpy())
            mixing_rows.append(output["mixing"].cpu().numpy())
            correction_rows.append(output["correction"].cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        condition_ids=np.asarray(condition_ids),
        hardening_parameters=np.concatenate(parameter_rows),
        hardening_radius=np.concatenate(hardening_rows),
        r_values=np.concatenate(r_rows),
        anchor_mixing=np.concatenate(mixing_rows),
        anchor_correction=np.concatenate(correction_rows),
    )


if __name__ == "__main__":
    main()

