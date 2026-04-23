# 🧬 Project 65: 4-Month GNN Sprint Tracker

## Month 1: The Foundation (Data & Connectivity)
- [x] Environment configured (M1 MPS, PyG, nilearn, monai).
- [x] Extracted baseline Adjacency Matrix using Schaefer-100.
- [x] Drafted initial Siamese GCN architecture.
- [ ] Receive official Twin Dataset from supervisor.
- [x] Write data preprocessing script to batch convert `.nii` to `Adjacency Matrices`. (`scripts/preprocess_twins.py`)
- [x] Finalize `TwinBrainDataset` class for PyTorch DataLoader. (`src/utils/brain_dataset.py` + `twin_collate`)
- [x] Build synthetic twin dataset generator with tunable ground-truth h^2. (`scripts/generate_synthetic_twins.py`)
- [x] Harden `SiameseBrainNet` (projection head, BatchNorm, edge-weight handling, config dataclass).
- [x] Write spectral graph theory primer for dissertation + grant. (`docs/spectral_primer.md`)

## Month 2: The Architecture (Multimodal Siamese GNN)
- [x] Connect `TwinBrainDataset` to the `SiameseBrainNet`. (`twin_collate` + `src/training/trainer.py`)
- [x] Verify Contrastive Loss logic with a small batch of real data. (smoke_test_model.py)
- [x] Build family-stratified K-fold cross-validation. (`src/analysis/splits.py`)
- [x] Write full training loop (AdamW + cosine LR + early stop + TensorBoard). (`scripts/train.py`)
- [x] Generate grant-figure h^2 recovery sweep on synthetic data. (`data/h2_sweep/h2_recovery.png`)
- [x] Implement Genetic Data fusion (MLP GeneticEncoder + concat/gated fusion). (`MultimodalSiameseBrainNet` in `src/models/siamese_gnn.py`)
- [x] Heritability auxiliary loss: `λ · MSE(\hat{h}^2_{batch}, h^2_{target})`. (`HeritabilityAuxLoss` in `src/models/siamese_gnn.py`, wired through `TrainConfig`)
- [x] Full pipeline integration test with TensorBoard monitoring. (`scripts/full_pipeline_test.py`)
- [x] Comprehensive README with usage guide and metric interpretation. (`README.md`)
- [ ] Ensure model trains without crashing (Memory profiling on M1/Cloud).

## Month 3: The Science (Training & Heritability)
- [ ] Execute full training loop on HPC / Cloud GPU.
- [ ] Tune hyperparameters (Margin, Learning Rate, GCN Hidden Channels).
- [x] Calculate Heritability ($h^2$) estimates based on twin similarities. (`src/analysis/heritability.py` — Falconer + bootstrap CI)
- [x] Generate t-SNE / UMAP plots of the latent space (showing MZ twins clustering). (`scripts/plot_latent_space.py` + `data/h2_sweep/latent_space_h100.png`)

## Month 4: The Evaluation & Report
- [~] Perform Ablation Studies (Graph-only vs. Genetics-only vs. Multimodal). — *Graph-only vs. Multimodal+aux sweep complete on synthetic data (`data/h2_sweep/sweep_results.csv`). Still need: Genetics-only variant + real-data ablation.*
- [ ] Compare GNN performance against a baseline CNN or ACE statistical model.
- [ ] Finalize visualizations (3D brain connectivity maps).
- [ ] Draft final dissertation report and submit.