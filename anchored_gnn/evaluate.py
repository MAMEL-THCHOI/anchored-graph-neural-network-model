"""Evaluate saved predictions with R-squared and error metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .workflow import load_targets, metric, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--group", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    split_map = read_split(args.split)
    target_index = {str(value): index for index, value in enumerate(targets["condition_ids"])}
    with np.load(args.predictions, allow_pickle=False) as source:
        prediction_ids = [str(value) for value in source["condition_ids"]]
        prediction_index = {value: index for index, value in enumerate(prediction_ids)}
        predicted_hardening = np.asarray(source["hardening_radius"])
        predicted_r = np.asarray(source["r_values"])
    ids = [value for value, group in split_map.items() if group == args.group]
    missing = [value for value in ids if value not in target_index or value not in prediction_index]
    if missing:
        raise ValueError(f"missing conditions: {missing[:5]}")
    actual_rows = np.asarray([target_index[value] for value in ids])
    predicted_rows = np.asarray([prediction_index[value] for value in ids])
    result = {
        "group": args.group,
        "hardening": metric(targets["hardening_radius"][actual_rows], predicted_hardening[predicted_rows]),
        "r": metric(targets["r_values"][actual_rows], predicted_r[predicted_rows]),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
