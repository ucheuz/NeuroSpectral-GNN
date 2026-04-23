# NeuroSpectral-GNN

**Project 65: Decoding Genetic vs. Environmental Brain Phenotype with Multimodal Graph Neural Networks**

A Siamese Graph Neural Network for estimating the heritability (h²) of brain connectivity patterns from twin fMRI data. This project uses spectral graph convolutions on functional connectomes to learn genetically-informed representations, where monozygotic (MZ) twin pairs cluster more tightly than dizygotic (DZ) pairs.

## Overview

The core idea: if genetics influence brain connectivity, then MZ twins (who share 100% of their DNA) should have more similar brain networks than DZ twins (who share ~50%). We train a Siamese GNN with contrastive loss to learn this structure, then estimate heritability using Falconer's formula on the learned embeddings.

**Key features:**
- Spectral graph convolutions (GCNConv) on brain connectivity graphs
- Multimodal fusion of connectome + polygenic risk scores (PRS)
- Heritability-aware auxiliary loss for calibrated h² estimation
- Family-stratified cross-validation to prevent data leakage
- Comprehensive synthetic data generation for validation

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n neurognn python=3.10
conda activate neurognn

# Install PyTorch (M1 Mac)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt

# Install TensorBoard for monitoring
pip install tensorboard
```

### 2. Run the Full Pipeline Test

```bash
# Quick test (~2 minutes)
python scripts/full_pipeline_test.py --output-dir data/pipeline_test --quick

# Full test (~5 minutes)
python scripts/full_pipeline_test.py --output-dir data/pipeline_test --clean
```

This runs:
1. Smoke tests (preprocessing + model)
2. Synthetic cohort generation
3. Graph-only baseline training
4. Multimodal + aux-loss training
5. h² recovery sweep
6. Latent-space visualizations

### 3. Monitor Training with TensorBoard

```bash
tensorboard --logdir data/pipeline_test/tensorboard
# Open http://localhost:6006
```

---

## Understanding the TensorBoard Metrics

### Loss Metrics

| Metric | What it shows | Good pattern |
|--------|---------------|--------------|
| `loss/train` | Training loss per epoch | Smooth downward curve |
| `loss/val` | Validation loss (triggers early stopping) | Decreases then flattens |
| `loss/train_aux_h2` | Heritability calibration loss (multimodal only) | Drops toward 0 |

**What to watch for:**
- If `train` drops but `val` rises → **overfitting**
- If both are flat → **model isn't learning** (check learning rate)
- If `train_aux_h2` oscillates wildly → **batch size too small**

### Validation Metrics

| Metric | What it shows | Target |
|--------|---------------|--------|
| `val/h2` | Estimated heritability from embeddings | Should match ground truth |
| `val/auc` | MZ vs DZ classification accuracy | Higher = better discrimination |
| `val/mean_mz_distance` | Average distance between MZ twin embeddings | Should be small |
| `val/mean_dz_distance` | Average distance between DZ twin embeddings | Should be larger than MZ |

**The key insight:** The gap between MZ and DZ distances IS the learned genetic signal.

### Comparing Models

Select multiple runs in the TensorBoard sidebar to overlay them:

| Question | Compare | Good sign |
|----------|---------|-----------|
| Does PRS help? | `val/auc` across models | Multimodal ≥ graph-only |
| Is aux-loss calibrating? | `val/h2` | Closer to ground truth |
| Which generalizes better? | `loss/val` | Lower final value |

### Expected Patterns by Heritability Level

| Ground truth h² | Expected AUC | Expected behavior |
|-----------------|--------------|-------------------|
| 0.0 | ~0.5 | Random (no genetic signal) |
| 0.4–0.6 | 0.6–0.8 | Moderate separation |
| 1.0 | ~1.0 | Perfect separation, MZ distance → 0 |

---

## Project Structure

```
NeuroSpectral-GNN/
├── src/
│   ├── models/
│   │   └── siamese_gnn.py      # BrainGNNEncoder, SiameseBrainNet, MultimodalSiameseBrainNet
│   ├── preprocessing/
│   │   ├── atlas.py            # Schaefer atlas loading
│   │   ├── connectivity.py     # Timeseries extraction, Fisher-z correlation
│   │   ├── graph.py            # Connectivity → PyG Data conversion
│   │   ├── manifest.py         # Twin pair metadata parsing
│   │   ├── pipeline.py         # Full preprocessing orchestration
│   │   └── synthetic.py        # Synthetic twin data generation
│   ├── training/
│   │   └── trainer.py          # Training loop, CV, early stopping
│   ├── analysis/
│   │   ├── heritability.py     # Falconer h², HeritabilityHead
│   │   └── splits.py           # Family-stratified K-fold
│   └── utils/
│       ├── brain_dataset.py    # TwinBrainDataset, twin_collate
│       ├── device.py           # MPS/CUDA/CPU selection
│       └── seeds.py            # Reproducibility
├── scripts/
│   ├── full_pipeline_test.py   # End-to-end validation
│   ├── train.py                # CLI for training
│   ├── run_h2_sweep.py         # Heritability recovery sweep
│   ├── plot_latent_space.py    # t-SNE/UMAP visualization
│   ├── generate_synthetic_twins.py
│   ├── preprocess_twins.py     # Real data preprocessing
│   └── smoke_test_*.py         # Quick validation scripts
├── docs/
│   └── spectral_primer.md      # Spectral graph theory explainer
├── data/                       # Generated during testing
└── requirements.txt
```

---

## CLI Reference

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_twins.py \
    --output-dir data/my_cohort \
    --n-mz 40 --n-dz 40 \
    --heritability 0.6 \
    --n-rois 100 \
    --prs-dim 16  # Enable PRS for multimodal
```

### Train a Model

```bash
# Graph-only
python scripts/train.py \
    --data-root data/my_cohort \
    --output-dir runs/experiment1 \
    --in-channels 100 \
    --max-epochs 50

# Multimodal with aux loss
python scripts/train.py \
    --data-root data/my_cohort \
    --output-dir runs/experiment2 \
    --in-channels 100 \
    --prs-dim 16 \
    --heritability-aux-weight 0.2 \
    --heritability-aux-target 0.6
```

### Run h² Sweep (Grant Figure)

```bash
python scripts/run_h2_sweep.py \
    --output-dir data/h2_sweep \
    --n-mz 60 --n-dz 60 \
    --max-epochs 40 \
    --save-checkpoints
```

Outputs:
- `h2_recovery.png` — Two-panel figure showing h² recovery and AUC
- `sweep_results.csv` — Raw data for methods section

### Generate Latent Space Figure

```bash
python scripts/plot_latent_space.py \
    --run-dir runs/experiment1 \
    --cohort-dir data/my_cohort \
    --output figures/latent_space.png
```

---

## Key Architectural Decisions

### Why Spectral Graph Convolutions?

GCNConv implements a first-order Chebyshev approximation of the spectral graph convolution:

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

This acts as a **low-pass filter on the graph spectrum** — smoothing node features according to graph structure. For brain connectomes, this captures the intuition that functionally connected regions should have similar representations.

See `docs/spectral_primer.md` for the full mathematical treatment.

### Why Contrastive Loss?

We frame heritability estimation as a metric learning problem:
- MZ twins (label=0) → minimize distance
- DZ twins (label=1) → push apart beyond margin

This learns a representation where genetic similarity maps to geometric proximity.

### Why the Auxiliary h² Loss?

Standard contrastive loss produces good **relative** rankings (MZ < DZ) but doesn't guarantee **calibrated absolute h²**. The auxiliary loss:

```
L_aux = λ · (ĥ²_batch - h²_target)²
```

directly supervises the batch-level heritability estimate, anchoring the learned manifold to the correct scale.

---

## Troubleshooting

| Problem | Symptom | Solution |
|---------|---------|----------|
| `torch.load` / `UnpicklingError` on `.pt` graphs | PyTorch 2.6+ defaults to `weights_only=True` | Use `torch.load(path, weights_only=False, map_location="cpu")` for PyG `Data` files (your own data only) |
| TensorBoard empty | "No dashboards active" | Install: `pip install tensorboard` |
| NaN loss | Loss becomes NaN | Already fixed — we use `abs(edge_weight)` for GCN normalization |
| AUC stuck at 0.5 | No MZ/DZ separation | Check labels, increase epochs |
| Overfitting | Train↓ Val↑ | Increase dropout, reduce model size |
| Slow on M1 | Not using MPS | Check `src/utils/device.py` selects MPS |

---

## Citation

If you use this code for research, please cite:

```
@mastersthesis{neurospectral2026,
  title={Decoding Genetic vs. Environmental Brain Phenotype with Multimodal Graph Neural Networks},
  author={[Kosiasochukwu Uzoka]},
  school={King's College London},
  year={2026}
}
```

---

## License

MIT License — see LICENSE file for details.
