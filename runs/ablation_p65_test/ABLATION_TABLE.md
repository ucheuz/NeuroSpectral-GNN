# P65 ablation (Siamese twin GNN)


- Data: `data/p65_quicktest/cohort_graph_only`
- in_channels: 64 (modality blocks: (16, 16, 16, 16))
- device_preference: `mps`

| Variant | best val loss (mean over folds) | AUC (mean) | pair accuracy (mean) |
| --- | ---: | ---: | ---: |
| `graph_only_concat` | 0.2062 | 0.6850 | 0.6250 |
| `attention_node_mlp_no_gcn` | 0.2879 | 0.7400 | 0.7000 |
| `full_mha_gcn` | 0.2758 | 0.7200 | 0.6000 |
