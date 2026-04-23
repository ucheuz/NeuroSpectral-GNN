#!/usr/bin/env python
"""Run P65 GNN ablations: graph-only, attention-only, full; write Markdown + JSON.

    python scripts/run_ablation.py --data-root data/synthetic_h060 --output-dir runs/ablation_p65

Uses ``device_preference`` (default **mps**); see ``get_device`` for fallbacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import TrainConfig, run_cross_validation  # noqa: E402


def _split_in_channels(n: int, m: int) -> tuple[int, ...]:
    if n % m != 0:
        raise SystemExit(
            f"in-channels {n} must be divisible by modality-parts {m} for this helper"
        )
    k = n // m
    return tuple([k] * m)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "ablation_p65")
    p.add_argument("--in-channels", type=int, default=100)
    p.add_argument(
        "--modality-parts",
        type=int,
        default=4,
        help="Split in_channels into M equal parts for the fusion block.",
    )
    p.add_argument(
        "--modality-names",
        nargs="*",
        default=[],
        help="Optional M names (first M values used; else M1, M2, …).",
    )
    p.add_argument("--include-zygosities", nargs="+", default=["MZ", "DZ"])
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--cross-modal-d-model", type=int, default=64)
    p.add_argument("--cross-modal-num-heads", type=int, default=4)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-checkpoints", action="store_true")
    args = p.parse_args()

    parts = _split_in_channels(args.in_channels, args.modality_parts)
    m = len(parts)
    if len(args.modality_names) >= m:
        mnames: tuple[str, ...] = tuple(args.modality_names[i] for i in range(m))
    else:
        mnames = tuple(f"M{i+1}" for i in range(m))

    base = TrainConfig(
        data_root=Path(args.data_root),
        output_dir=Path(args.data_root) / "_tmp_ablation",  # overwritten per run
        include_zygosities=tuple(args.include_zygosities),
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        prs_dim=0,
        model_type="graph",
        modality_feature_dims=parts,
        modality_names=mnames,
        cross_modal_d_model=args.cross_modal_d_model,
        cross_modal_num_heads=args.cross_modal_num_heads,
        device_preference=args.device,
        seed=args.seed,
        tensorboard=not args.no_tensorboard,
        save_checkpoints=not args.no_checkpoints,
    )

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, bool, bool]] = [
        ("graph_only_concat", False, False),
        ("attention_node_mlp_no_gcn", True, True),
        ("full_mha_gcn", True, False),
    ]
    table_rows: list[tuple[str, float, float, float]] = []

    for name, use_mha, skip_g in variants:
        odir = out_root / name
        odir.mkdir(parents=True, exist_ok=True)
        cfg = replace(
            base,
            output_dir=odir,
            use_cross_modal_attention=use_mha,
            skip_graph_conv=skip_g,
        )
        (odir / "ablation_name.txt").write_text(name)
        _ = run_cross_validation(cfg)
        summ = json.loads((odir / "cv_summary.json").read_text())
        mloss = float(
            np.mean([f["best_val_loss"] for f in summ["per_fold"]])
        )
        table_rows.append(
            (
                name,
                mloss,
                float(summ["mean_auc"]),
                float(summ.get("mean_pair_accuracy", float("nan"))),
            )
        )

    lines = [
        "# P65 ablation (Siamese twin GNN)\n",
        "",
        f"- Data: `{args.data_root}`",
        f"- in_channels: {args.in_channels} (modality blocks: {parts!r})",
        f"- device_preference: `{args.device}`",
        "",
        "| Variant | best val loss (mean over folds) | AUC (mean) | pair accuracy (mean) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, mloss, auc, acc in table_rows:
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "nan"
        lines.append(
            f"| `{name}` | {mloss:.4f} | {auc:.4f} | {acc_s} |"
        )
    report = out_root / "ABLATION_TABLE.md"
    report.write_text("\n".join(lines) + "\n")
    (out_root / "ablation_table.json").write_text(
        json.dumps(
            [
                {
                    "variant": name,
                    "mean_best_val_loss": mloss,
                    "mean_auc": auc,
                    "mean_pair_accuracy": acc,
                }
                for name, mloss, auc, acc in table_rows
            ],
            indent=2,
        )
    )
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
