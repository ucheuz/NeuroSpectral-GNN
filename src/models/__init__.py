from src.models.genetics_encoder import GeneticsEncoder
from src.models.siamese_gnn import (
    BrainGNN,
    BrainGNNEncoder,
    ModalityCrossAttentionBlock,
    ContrastiveLoss,
    GatedFusion,
    GeneticsOnlySiameseNet,
    HeritabilityAuxLoss,
    MultimodalSiameseBrainNet,
    ProjectionHead,
    SiameseBrainNet,
    SiameseConfig,
    TwinBatch,
    build_siamese_model,
)

# Backwards compatibility (old name).
GeneticEncoder = GeneticsEncoder

__all__ = [
    "BrainGNN",
    "BrainGNNEncoder",
    "ModalityCrossAttentionBlock",
    "ContrastiveLoss",
    "GatedFusion",
    "GeneticEncoder",
    "GeneticsEncoder",
    "GeneticsOnlySiameseNet",
    "HeritabilityAuxLoss",
    "MultimodalSiameseBrainNet",
    "ProjectionHead",
    "SiameseBrainNet",
    "SiameseConfig",
    "TwinBatch",
    "build_siamese_model",
]
