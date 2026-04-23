# Spectral Graph Theory Primer for Project 65

> **Purpose.** A self-contained intuition-first walk-through of *why* we use a
> Graph Neural Network (and specifically `GCNConv`) to analyse brain
> connectomes. Written so chunks can be lifted verbatim into the dissertation
> introduction and the £30 k Seed Award "Technical Approach" section.

---

## 1. The one-paragraph version (for grant reviewers)

A functional brain connectome is a graph: nodes are Schaefer parcels, edges
are Fisher-z Pearson correlations between their BOLD timeseries. This graph
lives in a non-Euclidean space — it has no canonical coordinate system, no
shift operator, and no notion of "neighbouring pixels." Traditional CNNs
(which assume a regular Euclidean grid) are therefore structurally
inappropriate. **Spectral Graph Theory** provides the alternative: the
eigendecomposition of the graph Laplacian `L = D − A` defines a graph-native
Fourier transform, which in turn defines a *graph convolution*. Modern Graph
Neural Networks (in particular `GCNConv`, Kipf & Welling 2017) implement an
efficient, localised approximation of this spectral convolution, yielding a
principled inductive bias: the model is forced to learn smoothly-varying
functions across the connectome rather than memorise arbitrary
subject-specific wiring. This is the mathematical foundation on which our
Siamese heritability analysis rests.

---

## 2. From images to graphs: why `Conv2d` fails

A classical image CNN applies the same small filter at every spatial
location. This works because images live on a **regular grid** with a fixed,
translation-equivariant neighbourhood structure. A 3x3 kernel at pixel
`(i, j)` always sees the same eight neighbours.

A brain graph has none of that structure:

| Property                 | Image grid      | Brain graph                |
|--------------------------|-----------------|----------------------------|
| Node neighbourhood size  | constant (8)    | variable (degree varies)   |
| Node ordering            | canonical (i,j) | arbitrary                  |
| Shift operator           | pixel shift     | none                       |
| "Convolution" meaning    | filter dot grid | must be redefined spectrally |

To convolve on a graph we need a new definition — and that definition comes
from the Laplacian's spectrum.

---

## 3. The Graph Laplacian, informally

Let `A` be the N×N adjacency matrix of the brain graph (with edge weights).
Let `D = diag(degree_i)` be the diagonal degree matrix. Two canonical
Laplacians:

- **Combinatorial**:  `L = D − A`
- **Symmetric normalised**:  `L_sym = I − D^{-1/2} A D^{-1/2}`

`L_sym` is what `GCNConv` uses under the hood. Intuitively, for any function
`f : nodes → ℝ` (think: activation of each ROI), the quantity

    (L f)_i = Σ_{j ∼ i} A_ij · (f_i − f_j)

measures how much `f` disagrees with its neighbours at node `i`. It is the
graph analogue of the Laplacian operator `∇²` on continuous domains — which
in physics governs diffusion, heat flow, and wave propagation.

**Why that matters for brains.** If we think of functional coupling as
something that "diffuses" along white-matter and functional connections, then
the Laplacian literally encodes the diffusion operator of the connectome.
Learning a function of `L` is learning a function of brain communication
dynamics.

---

## 4. The Graph Fourier Transform (GFT)

`L_sym` is real, symmetric, and positive-semidefinite. So by the spectral
theorem it has an orthonormal eigendecomposition:

    L_sym = U Λ U^T,   Λ = diag(λ_1, ..., λ_N),   U^T U = I.

The eigenvalues `λ_k ∈ [0, 2]` (for `L_sym`) are the **graph frequencies**;
the eigenvectors `u_k` (columns of `U`) are the **graph Fourier modes**.

- `λ_1 = 0` and `u_1 ∝ 1` — the constant "DC" mode.
- Small `λ`: smooth modes (neighbouring ROIs take similar values). Think
  resting-state networks.
- Large `λ`: oscillatory modes, sign-flips across edges. Think high-frequency
  noise or rare switching patterns.

Given a signal `f ∈ ℝ^N` on the nodes (e.g. a column of our connectivity
matrix), its **Graph Fourier Transform** is

    f̂ = U^T f.

Each entry `f̂_k` tells us how much of mode `u_k` is present in `f`. Exactly
analogous to the classical Fourier transform for periodic signals.

---

## 5. Spectral graph convolution

Classical convolution on the real line factorises as multiplication in the
Fourier domain:

    (f * g)(x) = IFT( FT(f) · FT(g) )

Transplanting that to graphs:

    f *_G g_θ := U · diag( ĝ_θ(λ_1), ..., ĝ_θ(λ_N) ) · U^T · f      ...  (*)

where `ĝ_θ` is a learnable filter function of the eigenvalues. This is the
**Bruna et al. 2014** formulation. It's mathematically beautiful but
computationally ugly: computing `U` costs O(N^3) and destroys graph locality.

### 5.1 Polynomial (Chebyshev) approximation — Defferrard 2016

Restrict `ĝ_θ(λ)` to a polynomial of degree `K` in `λ`:

    ĝ_θ(λ) = Σ_{k=0}^{K} θ_k · T_k(λ̃),   λ̃ = 2λ/λ_max − 1.

Plugging into (*) and using `L = U Λ U^T`, the convolution becomes

    f *_G g_θ = Σ_{k=0}^{K} θ_k · T_k(L̃) · f.

No eigendecomposition needed — just repeated sparse matrix-vector products
with `L̃`. K-localised: each layer only mixes information within `K` hops.

### 5.2 First-order simplification — Kipf & Welling 2017 (`GCNConv`)

Set `K = 1` and absorb scaling into trainable weights `W`:

    H^{(l+1)} = σ( D̂^{-1/2} Â D̂^{-1/2} · H^{(l)} · W^{(l)} )                ...  (GCN)

where `Â = A + I` (self-loops), `D̂` the corresponding degree matrix, and
`σ` a non-linearity. Every `GCNConv` layer in our Siamese encoder executes
exactly this.

**Interpreted spectrally**, each GCN layer is a **low-pass filter** on the
graph: the filter response in the Fourier domain is approximately
`ĝ(λ) = 1 − λ`, which attenuates high-frequency (high-`λ`) modes. This is
the inductive bias that makes GCNs so effective on real-world graphs where
the useful signal is smooth across neighbourhoods.

> **Grant-proposal phrasing:**
> *"The proposed architecture exploits the symmetric normalised Laplacian of
> each subject's connectome as an implicit low-pass filter in the graph
> Fourier domain (Kipf & Welling, 2017), a principled inductive bias that
> favours smoothly varying functional organisation - precisely the signal
> we expect shared genetic architecture to manifest across monozygotic twins."*

---

## 6. Why this matters for heritability estimation

Two concrete reasons the spectral view earns its place in this project:

1. **Graph-aware similarity.** Two brain graphs that are isomorphic but
   have ROIs shuffled look identical under a spectral GNN (permutation
   equivariance is baked in). Two brain graphs that happen to share a few
   high-magnitude edges but are globally different look *different*. This
   is the correct notion of similarity for heritability — we want shared
   *organisation*, not shared *labels*.

2. **Smoothness prior aligns with neuroscience.** Genetic effects on brain
   connectivity are expected to manifest as coherent modulations of
   canonical networks (Yeo 7 / 17), not random edge-wise perturbations. The
   low-pass bias of `GCNConv` prefers such solutions automatically,
   reducing overfitting on the small MZ+DZ cohort.

---

## 7. Reading the spectrum of a real brain graph

The code below (runnable with our preprocessed synthetic cohort) produces
Figure S1 for the dissertation: the eigenvalue distribution of an MZ twin's
normalised Laplacian compared to a DZ twin's — the *spectral fingerprint*.

```python
import numpy as np
import torch
from scipy.linalg import eigh
import matplotlib.pyplot as plt

data = torch.load("data/synthetic_h060/subjects/FAM_MZ_0000_A.pt",
                  weights_only=False)
A = data.connectivity.numpy()
A = np.abs(A)  # treat as unsigned for spectral analysis
np.fill_diagonal(A, 0.0)
D = np.diag(A.sum(axis=1))
L_sym = np.eye(A.shape[0]) - np.linalg.inv(np.sqrt(D + 1e-8)) @ A @ np.linalg.inv(np.sqrt(D + 1e-8))
eigvals = eigh(L_sym, eigvals_only=True)

plt.hist(eigvals, bins=30, edgecolor="k")
plt.xlabel("Eigenvalue (graph frequency)"); plt.ylabel("Count")
plt.title("Normalised Laplacian spectrum — MZ twin A")
plt.show()
```

> **MPS tip.** Never call `torch.linalg.eigh` on MPS - it falls back to CPU
> and is slow. Always move the tensor to CPU first:
>
>     eigvals, eigvecs = torch.linalg.eigh(L.cpu())

---

## 8. References (for the dissertation bibliography)

- Bruna, J., Zaremba, W., Szlam, A., & LeCun, Y. (2014). *Spectral Networks
  and Locally Connected Networks on Graphs*. ICLR.
- Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). *Convolutional
  Neural Networks on Graphs with Fast Localized Spectral Filtering*. NeurIPS.
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with
  Graph Convolutional Networks*. ICLR.
- Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst,
  P. (2013). *The Emerging Field of Signal Processing on Graphs*. IEEE SPM.
- Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P.
  (2017). *Geometric Deep Learning: Going Beyond Euclidean Data*. IEEE SPM.
- Schaefer, A. et al. (2018). *Local-Global Parcellation of the Human
  Cerebral Cortex*. Cerebral Cortex.
- Li, X. et al. (2021). *BrainGNN: Interpretable Brain Graph Neural Network
  for fMRI Analysis*. Medical Image Analysis.
- Falconer, D. S. (1960). *Introduction to Quantitative Genetics*.

---

## 9. TL;DR cheat-sheet

| Concept                  | Classical analogue     | Graph version                       |
|--------------------------|------------------------|-------------------------------------|
| Domain                   | ℝ^2 grid               | vertices V of a graph               |
| Shift operator           | translation            | adjacency multiplication            |
| Laplacian                | ∇² = ∂²/∂x² + ∂²/∂y²   | `L = D − A` / `L_sym = I − D^{-1/2} A D^{-1/2}` |
| Fourier basis            | complex exponentials   | eigenvectors of L                    |
| Fourier transform        | `F f = ∫ f · e^{-ikx}` | `f̂ = U^T f`                         |
| Convolution              | `f * g = IFT(FT f · FT g)` | `U · diag(ĝ(Λ)) · U^T · f`       |
| Efficient approx         | FFT                    | Chebyshev polynomials / GCNConv      |
| Inductive bias           | translation equivariance | permutation equivariance + low-pass |
