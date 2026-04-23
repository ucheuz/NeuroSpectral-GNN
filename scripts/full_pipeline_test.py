#!/usr/bin/env python
"""Full pipeline integration test with TensorBoard monitoring.

Runs the complete NeuroSpectral-GNN pipeline on synthetic data:
    1. Smoke tests (preprocessing + model)
    2. Generate synthetic cohorts (optional UNREL pairs)
    3. Train graph-only baseline
    4. Train multimodal + aux-loss variant
    5. Train genetics-only (PRS ablation)
    6. Run h² recovery sweep (generates grant figure)
    7. Latent-space + 3D connectome figures
    8. Print pass/fail summary

TensorBoard logs are written to {output_dir}/tensorboard/ — launch with:
    tensorboard --logdir {output_dir}/tensorboard

Usage
-----
    python scripts/full_pipeline_test.py --output-dir data/pipeline_test

    # Quick mode (smaller cohorts, fewer epochs — ~2 min)
    python scripts/full_pipeline_test.py --output-dir data/pipeline_test --quick

    # Verbose mode (see all subprocess output)
    python scripts/full_pipeline_test.py --output-dir data/pipeline_test --verbose
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable  # Use the same Python that's running this script


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float
    message: str = ""
    output_path: Optional[Path] = None


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {result.name} ({result.duration_s:.1f}s)")
        if result.message:
            print(f"         {result.message}")

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines = [
            "",
            "=" * 60,
            f"  PIPELINE TEST SUMMARY: {passed}/{total} passed",
            "=" * 60,
        ]
        for r in self.results:
            icon = "\033[92m✓\033[0m" if r.passed else "\033[91m✗\033[0m"
            lines.append(f"  {icon} {r.name}")
            if r.output_path and r.output_path.exists():
                lines.append(f"      -> {r.output_path}")
        lines.append("")
        return "\n".join(lines)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


def _run(
    cmd: list[str],
    name: str,
    verbose: bool,
    cwd: Path = REPO_ROOT,
    output_path: Optional[Path] = None,
) -> TestResult:
    """Run a subprocess and return a TestResult."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=not verbose,
            text=True,
            check=True,
        )
        duration = time.perf_counter() - t0
        return TestResult(
            name=name,
            passed=True,
            duration_s=duration,
            output_path=output_path,
        )
    except subprocess.CalledProcessError as e:
        duration = time.perf_counter() - t0
        msg = ""
        if e.stderr:
            # Last 3 lines of stderr
            msg = "\n".join(e.stderr.strip().split("\n")[-3:])
        return TestResult(
            name=name,
            passed=False,
            duration_s=duration,
            message=msg or f"Exit code {e.returncode}",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pipeline_test"),
        help="Root directory for all test outputs",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: smaller cohorts, fewer epochs (~2 min total)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show full subprocess output (useful for debugging)",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke tests (useful if you've already run them)",
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the h² sweep (the longest step)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove output-dir before starting (fresh run)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.output_dir.resolve()
    suite = TestSuite()

    # Config based on --quick flag
    if args.quick:
        n_mz, n_dz = 20, 20
        n_unrel = 4
        max_epochs = 15
        n_splits = 2
        sweep_h2 = [0.0, 0.5, 1.0]
    else:
        n_mz, n_dz = 40, 40
        n_unrel = 10
        max_epochs = 30
        n_splits = 3
        sweep_h2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    n_rois = 64
    prs_dim = 16
    zyg_train = ["MZ", "DZ", "UNREL"] if n_unrel > 0 else ["MZ", "DZ"]

    print("\n" + "=" * 60)
    print("  NeuroSpectral-GNN Full Pipeline Test")
    print("=" * 60)
    print(f"  Output directory : {out}")
    print(f"  Mode             : {'quick' if args.quick else 'full'}")
    print(f"  Cohort size      : {n_mz} MZ + {n_dz} DZ + {n_unrel} UNREL pairs")
    print(f"  Max epochs       : {max_epochs}")
    print(f"  h² sweep points  : {sweep_h2}")
    print("=" * 60 + "\n")

    if args.clean and out.exists():
        print(f"  Cleaning {out}...")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tb_dir = out / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    print(f"\n  \033[96mTensorBoard logs: {tb_dir}\033[0m")
    print(f"  \033[96mLaunch with: tensorboard --logdir {tb_dir}\033[0m\n")

    # -------------------------------------------------------------------------
    # 1. Smoke tests
    # -------------------------------------------------------------------------
    if not args.skip_smoke:
        print("\n[1/8] Running smoke tests...\n")

        suite.add(
            _run(
                [PYTHON, "scripts/smoke_test_preprocess.py"],
                name="Preprocessing smoke test",
                verbose=args.verbose,
            )
        )

        suite.add(
            _run(
                [PYTHON, "scripts/smoke_test_model.py"],
                name="Model smoke test (graph + multimodal)",
                verbose=args.verbose,
            )
        )
    else:
        print("\n[1/8] Skipping smoke tests (--skip-smoke)\n")

    # -------------------------------------------------------------------------
    # 2. Generate synthetic cohorts
    # -------------------------------------------------------------------------
    print("\n[2/8] Generating synthetic cohorts...\n")

    cohort_graph = out / "cohort_graph_only"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/generate_synthetic_twins.py",
                "--output-dir", str(cohort_graph),
                "--n-mz", str(n_mz),
                "--n-dz", str(n_dz),
                "--n-rois", str(n_rois),
                "--heritability", "0.6",
                "--n-unrelated", str(n_unrel),
                "--seed", "42",
            ],
            name="Generate graph-only cohort (h²=0.6)",
            verbose=args.verbose,
            output_path=cohort_graph,
        )
    )

    cohort_mm = out / "cohort_multimodal"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/generate_synthetic_twins.py",
                "--output-dir", str(cohort_mm),
                "--n-mz", str(n_mz),
                "--n-dz", str(n_dz),
                "--n-rois", str(n_rois),
                "--heritability", "0.7",
                "--prs-dim", str(prs_dim),
                "--n-unrelated", str(n_unrel),
                "--seed", "42",
            ],
            name="Generate multimodal cohort (h²=0.7, PRS)",
            verbose=args.verbose,
            output_path=cohort_mm,
        )
    )

    # -------------------------------------------------------------------------
    # 3. Train graph-only baseline
    # -------------------------------------------------------------------------
    print("\n[3/8] Training graph-only baseline...\n")

    run_graph = out / "runs" / "graph_only"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/train.py",
                "--data-root", str(cohort_graph),
                "--output-dir", str(run_graph),
                "--in-channels", str(n_rois),
                "--hidden-channels", "32",
                "--projection-dim", "16",
                "--max-epochs", str(max_epochs),
                "--n-splits", str(n_splits),
                "--batch-size", "8",
                "--patience", "8",
                "--include-zygosities", *zyg_train,
                "--seed", "42",
            ],
            name="Train graph-only model",
            verbose=args.verbose,
            output_path=run_graph / "cv_summary.json",
        )
    )

    # Copy TensorBoard logs to central location
    for fold_dir in run_graph.glob("fold_*"):
        tb_fold = fold_dir / "tb"  # trainer writes to 'tb/', not 'tensorboard/'
        if tb_fold.exists():
            dest = tb_dir / "graph_only" / fold_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in tb_fold.iterdir():
                shutil.copy(f, dest)

    # -------------------------------------------------------------------------
    # 4. Train multimodal + aux loss
    # -------------------------------------------------------------------------
    print("\n[4/8] Training multimodal + aux-loss model...\n")

    run_mm = out / "runs" / "multimodal_aux"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/train.py",
                "--data-root", str(cohort_mm),
                "--output-dir", str(run_mm),
                "--in-channels", str(n_rois),
                "--hidden-channels", "32",
                "--projection-dim", "16",
                "--prs-dim", str(prs_dim),
                "--prs-hidden", "32",
                "--prs-embed-dim", "32",
                "--heritability-aux-weight", "0.2",
                "--heritability-aux-target", "0.7",
                "--max-epochs", str(max_epochs),
                "--n-splits", str(n_splits),
                "--batch-size", "8",
                "--patience", "8",
                "--include-zygosities", *zyg_train,
                "--seed", "42",
            ],
            name="Train multimodal + aux-loss model",
            verbose=args.verbose,
            output_path=run_mm / "cv_summary.json",
        )
    )

    # Copy TensorBoard logs
    for fold_dir in run_mm.glob("fold_*"):
        tb_fold = fold_dir / "tb"
        if tb_fold.exists():
            dest = tb_dir / "multimodal_aux" / fold_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in tb_fold.iterdir():
                shutil.copy(f, dest)

    # -------------------------------------------------------------------------
    # 5. Train genetics-only (PRS ablation — completes ablation triangle)
    # -------------------------------------------------------------------------
    print("\n[5/8] Training genetics-only (PRS) baseline...\n")

    run_geo = out / "runs" / "genetics_only"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/train.py",
                "--data-root", str(cohort_mm),
                "--output-dir", str(run_geo),
                "--in-channels", str(n_rois),
                "--hidden-channels", "32",
                "--projection-dim", "16",
                "--prs-dim", str(prs_dim),
                "--prs-hidden", "32",
                "--prs-embed-dim", "32",
                "--model-type", "genetics_only",
                "--max-epochs", str(max_epochs),
                "--n-splits", str(n_splits),
                "--batch-size", "8",
                "--patience", "8",
                "--include-zygosities", *zyg_train,
                "--seed", "42",
            ],
            name="Train genetics-only (PRS) model",
            verbose=args.verbose,
            output_path=run_geo / "cv_summary.json",
        )
    )
    for fold_dir in run_geo.glob("fold_*"):
        tb_fold = fold_dir / "tb"
        if tb_fold.exists():
            dest = tb_dir / "genetics_only" / fold_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in tb_fold.iterdir():
                shutil.copy(f, dest)

    # -------------------------------------------------------------------------
    # 6. h² recovery sweep (grant figure)
    # -------------------------------------------------------------------------
    if not args.skip_sweep:
        print("\n[6/8] Running h² recovery sweep (this is the longest step)...\n")

        sweep_dir = out / "h2_sweep"
        h2_str = " ".join(str(h) for h in sweep_h2)
        suite.add(
            _run(
                [
                    PYTHON,
                    "scripts/run_h2_sweep.py",
                    "--output-dir", str(sweep_dir),
                    "--h2-values", *[str(h) for h in sweep_h2],
                    "--n-mz", str(n_mz),
                    "--n-dz", str(n_dz),
                    "--n-rois", str(n_rois),
                    "--max-epochs", str(max_epochs),
                    "--n-splits", str(n_splits),
                    "--batch-size", "8",
                    "--prs-dim", str(prs_dim),
                    "--aux-weight", "0.2",
                    "--n-unrelated", str(n_unrel),
                    "--save-checkpoints",
                    "--seed", "42",
                ],
                name=f"h² sweep ({len(sweep_h2)} points × 2 variants)",
                verbose=args.verbose,
                output_path=sweep_dir / "h2_recovery.png",
            )
        )

        # Copy sweep TensorBoard logs
        for variant_dir in (sweep_dir / "runs").glob("*"):
            for h_dir in variant_dir.glob("h_*"):
                for fold_dir in h_dir.glob("fold_*"):
                    tb_fold = fold_dir / "tb"
                    if tb_fold.exists():
                        dest = tb_dir / "sweep" / variant_dir.name / h_dir.name / fold_dir.name
                        dest.mkdir(parents=True, exist_ok=True)
                        for f in tb_fold.iterdir():
                            shutil.copy(f, dest)
    else:
        print("\n[6/8] Skipping h² sweep (--skip-sweep)\n")

    # -------------------------------------------------------------------------
    # 7. Latent-space visualizations
    # -------------------------------------------------------------------------
    print("\n[7/8] Generating latent-space visualizations...\n")

    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    # From the multimodal training run
    latent_mm = figures_dir / "latent_multimodal.png"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/plot_latent_space.py",
                "--run-dir", str(run_mm),
                "--cohort-dir", str(cohort_mm),
                "--output", str(latent_mm),
                "--seed", "42",
            ],
            name="Latent-space figure (multimodal h²=0.7)",
            verbose=args.verbose,
            output_path=latent_mm,
        )
    )

    # From the sweep h²=1.0 if available
    if not args.skip_sweep:
        sweep_h100 = out / "h2_sweep" / "runs" / "multimodal+aux" / "h_100"
        cohort_h100 = out / "h2_sweep" / "cohorts" / "h_100"
        if sweep_h100.exists() and cohort_h100.exists():
            latent_h100 = figures_dir / "latent_h100_sweep.png"
            suite.add(
                _run(
                    [
                        PYTHON,
                        "scripts/plot_latent_space.py",
                        "--run-dir", str(sweep_h100),
                        "--cohort-dir", str(cohort_h100),
                        "--output", str(latent_h100),
                        "--seed", "42",
                    ],
                    name="Latent-space figure (sweep h²=1.0)",
                    verbose=args.verbose,
                    output_path=latent_h100,
                )
            )

    # -------------------------------------------------------------------------
    # 8. 3D connectome (nilearn; grant supplementary figure)
    # -------------------------------------------------------------------------
    print("\n[8/8] 3D connectome render (illustrative)...\n")
    c3d = figures_dir / "connectome_3d.png"
    suite.add(
        _run(
            [
                PYTHON,
                "scripts/plot_brain_3d.py",
                "--n-rois", "100",
                "--output", str(c3d),
            ],
            name="3D connectome figure (Schaefer-100)",
            verbose=args.verbose,
            output_path=c3d,
        )
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(suite.summary())

    print(f"\n  \033[96mTo view training curves:\033[0m")
    print(f"  \033[96m  tensorboard --logdir {tb_dir}\033[0m")
    print(f"  \033[96m  Then open http://localhost:6006\033[0m\n")

    if suite.all_passed:
        print("  \033[92mAll tests passed! Pipeline is working correctly.\033[0m\n")
        return 0
    else:
        print("  \033[91mSome tests failed. Check output above for details.\033[0m\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
