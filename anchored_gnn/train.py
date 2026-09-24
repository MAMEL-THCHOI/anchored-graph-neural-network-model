"""Train the anchored grain graph model from user-supplied graph and response arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .model import AnchoredGrainGraph
from .workflow import (
    decode_parameters,
    feature_stats,
    fit_target_stats,
    fourier_basis,
    graph_items,
    joint_loss,
    load_raw_graphs,
    load_targets,
    make_loaders,
    metric,
    move_batch,
    normalized_target_tensors,
    radius_at_work,
    read_split,
    set_seed,
)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def validation_score(metrics: dict) -> float:
    """Sum of hardening and Lankford mean relative errors, in percent."""
    return 100.0 * (metrics["hardening"]["mre"] + metrics["r"]["mre"])


@torch.no_grad()
def evaluate_model(
    model: AnchoredGrainGraph,
    loader,
    targets: dict[str, np.ndarray],
    tensors: dict[str, torch.Tensor],
    target_stats: dict,
    basis: torch.Tensor,
    work_levels: np.ndarray,
    device: torch.device,
) -> dict[str, dict]:
    model.eval()
    hardening_rows, r_rows, index_rows = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        indices = batch["target_index"]
        lower = torch.log(tensors["anchor_lower"][indices].clamp(0.02, 100.0))
        upper = torch.log(tensors["anchor_upper"][indices].clamp(0.02, 100.0))
        output = model(batch, lower, upper)
        parameters = decode_parameters(output["coefficients"], basis, target_stats).cpu().numpy()
        hardening_rows.append(radius_at_work(parameters, work_levels))
        r_rows.append(torch.exp(torch.clamp(output["log_r"], -4.0, 5.0)).cpu().numpy())
        index_rows.append(indices.cpu().numpy())
    indices = np.concatenate(index_rows)
    hardening = np.concatenate(hardening_rows)
    r_values = np.concatenate(r_rows)
    return {
        "hardening": metric(targets["hardening_radius"][indices], hardening),
        "r": metric(targets["r_values"][indices], r_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(args.seed)
    device = torch.device(args.device)
    targets = load_targets(args.targets)
    condition_ids = targets["condition_ids"].tolist()
    target_index = {value: index for index, value in enumerate(condition_ids)}
    split_map = read_split(args.split)
    if set(condition_ids) != set(split_map):
        raise ValueError("target and split condition IDs must match exactly")

    raw_graphs = load_raw_graphs(args.graphs, condition_ids)
    train_ids = {value for value in condition_ids if split_map[value] == "train"}
    graph_stats = feature_stats(raw_graphs, train_ids)
    items = graph_items(raw_graphs, target_index, graph_stats)
    loaders = make_loaders(items, split_map, int(config["batch_size"]), args.seed)
    train_indices = np.asarray([target_index[value] for value in condition_ids if value in train_ids])
    target_stats = fit_target_stats(targets, train_indices)
    tensors = {key: value.to(device) for key, value in normalized_target_tensors(targets, target_stats).items()}
    basis = fourier_basis(int(config["fourier_order"])).to(device)
    model = AnchoredGrainGraph(
        coefficient_count=4 * basis.shape[1],
        lambda_init=float(target_stats["lambda_init"]),
        hidden=int(config["hidden"]),
        dropout=float(config["dropout"]),
        message_layers=int(config["message_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    work_levels = np.asarray(config["work_levels"], dtype=np.float64)
    best: tuple[float, int, str, dict[str, torch.Tensor] | None, dict | None] = (math.inf, 0, "", None, None)
    best_seen = 0
    history = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        epoch_losses = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            indices = batch["target_index"]
            lower = torch.log(tensors["anchor_lower"][indices].clamp(0.02, 100.0))
            upper = torch.log(tensors["anchor_upper"][indices].clamp(0.02, 100.0))
            output = model(batch, lower, upper)
            loss, parts = joint_loss(output, indices, tensors, target_stats, basis, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            epoch_losses.append({"total": float(loss.detach()), **parts})
        validation = evaluate_model(
            model, loaders["val"], targets, tensors, target_stats, basis, work_levels, device
        )
        score = validation_score(validation)
        if score < best[0] - 1.0e-9:
            best = (score, epoch, "raw", clone_state(model), validation)
            best_seen = epoch
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean([row["total"] for row in epoch_losses])),
                "hardening_loss": float(np.mean([row["hardening"] for row in epoch_losses])),
                "r_loss": float(np.mean([row["r"] for row in epoch_losses])),
                "auxiliary_loss": float(np.mean([row["auxiliary"] for row in epoch_losses])),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation_hardening_mre_percent": 100.0 * validation["hardening"]["mre"],
                "validation_r_mre_percent": 100.0 * validation["r"]["mre"],
                "validation_selection_score": score,
            }
        )
        if epoch - best_seen >= int(config["patience"]):
            break

    if best[3] is None:
        raise RuntimeError("training did not produce a checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best[3],
            "seed": args.seed,
            "selected_epoch": best[1],
            "selected_state_type": best[2],
            "config": config,
            "feature_stats": graph_stats,
            "target_stats": target_stats,
        },
        args.output / "model_best.pt",
    )
    summary = {
        "seed": args.seed,
        "selected_epoch": best[1],
        "selected_state_type": best[2],
        "validation": best[4],
        "epochs_completed": history[-1]["epoch"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

