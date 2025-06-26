# Cerebrum: A Multi-Mixture Model

# Table of Contents


1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Background and Related Work](#background-and-related-work)
4. [System Architecture](#system-architecture)

   1. [Overview](#overview)
   2. [Module Descriptions](#module-descriptions)

      1. [MM: Single-Model Wrapper](#mm-single-model-wrapper)
      2. [MMMan: Multi-Model Manager](#mmman-multi-model-manager)
      3. [Cerebrum: Top-Level Container](#cerebrum-top-level-container)
      4. [MultiMixtureTransformer: Hybrid Sequence Model](#multimixturetransformer-hybrid-sequence-model)
      5. [Supporting Components](#supporting-components)

         1. [Encoder and Decoder](#encoder-and-decoder)
         2. [TinyDiffusionBlock](#tinydiffusionblock)
         3. [DraftScorer](#draftscorer)
         4. [RecurrentNetwork and HiddenMarkov](#recurrentnetwork-and-hiddenmarkov)
         5. [GaussianMixture](#gaussianmixture)
         6. [TimeSeriesTransformer](#timeseriestransformer)
5. [Data Flow and Computational Graph](#data-flow-and-computational-graph)
6. [Training Procedures](#training-procedures)

   1. [GMM/HMM Maximum Likelihood](#gmmhmm-maximum-likelihood)
   2. [Hybrid Model Training](#hybrid-model-training)
   3. [Auxiliary Rule Losses](#auxiliary-rule-losses)
7. [Inference and Generation](#inference-and-generation)

   1. [Scoring and Sampling](#scoring-and-sampling)
   2. [Regression and Autoregressive Generation](#regression-and-autoregressive-generation)
8. [Performance Considerations](#performance-considerations)
9. [Example Usage Patterns](#example-usage-patterns)
10. [Component Summary Table](#component-summary-table)
11. [Conclusion and Future Work](#conclusion-and-future-work)
12. [References](#references)

# Abstract

We introduce **Cerebrum**, a modular framework unifying probabilistic mixture models (GMMs), state‐space models (HMMs), and deep generative architectures (VAEs, diffusion, and Transformers) into a cohesive platform. Cerebrum features three layers: (1) **MM**, a wrapper for single‐model fitting and evaluation; (2) **MMMan**, a manager of multiple GMM/HMM instances; and (3) **Cerebrum**, a top‐level `ModuleDict` for arbitrary submodels including the advanced **MultiMixtureTransformer** hybrid. We detail the design, data flow, training regimes, and inference mechanisms, and demonstrate utility on synthetic and real time‐series tasks.

# Introduction

Hybrid generative modeling has grown increasingly complex, combining components like Variational Autoencoders (VAEs), Gaussian Mixture Models (GMMs), Hidden Markov Models (HMMs), diffusion denoisers, and sequence Transformers. Integrating these disparate modules, each with unique parameterizations, loss functions, and inference algorithms, demands a unifying infrastructure. Cerebrum fulfills this need by providing:

* **Unified Interfaces**: consistent `.fit(...)`, `.score(...)`, and `.export_model(...)` across model types.
* **Hierarchical Management**: from single‐model MM to multi‐model MMMan to top‐level Cerebrum.
* **Composable Hybrids**: MultiMixtureTransformer that stitches VAEs, RNN‐HMM, GMM, diffusion blocks, and Transformers into one end‐to‐end trainable network.

# System Architecture

## Overview

Cerebrum layers:

1. **MM**: single‐model wrapper around GaussianMixture or HiddenMarkov. Handles tensor prep, training, scoring, and serialization.
2. **MMMan**: manages multiple MM instances, identified by integer (GMM) or string (HMM) IDs. Provides bulk operations over arbitrary subsets.
3. **Cerebrum**: top‐level container in a `ModuleDict`, supports arbitrary `nn.Module` submodels, including MMMan or MultiMixtureTransformer. Offers uniform API for fit/add, export/import, query, and assimilation of entire brains.

## Module Descriptions

### MM: Single-Model Wrapper

**Initialization**:

* Stores `model_type`, `n_features`, and either `self.gmm` or `self.hm`.
* Maintains `self.gmms` list and `self.hgmm_models` dict for fitted submodels.

**Key Methods**:

* `_prepare_tensor(X)`: casts input arrays to `torch.float32` Tensors, moves to correct device.
* `fit(...)`: gradient‐based MLE via Adam. For GMM: minimizes `-mean(log_prob)`. For HMM: minimizes `-log_prob(sequence)` per sequence, averaged.
* `unfit(data_id)`: remove by index or key.
* `check_data()`: returns mapping of stored IDs to type.
* `score(X)`: returns average log likelihood.
* `get_log_likelihoods(X)`: per-sample sequence or vector log-likelihoods.
* `get_means()`, `get_variances()`, `get_weights()`: expose learned parameters as NumPy arrays.
* `export_model(filepath)`, `import_model(source)`: state‐dict serialization.

### MMMan: Multi-Model Manager

**Initialization**: empty lists/dicts.

**Unique ID Generation**: random 6‐letter strings for HMMs, integer indices for GMMs.

**fit(data, ...)**:

1. If `data` is instance of `GaussianMixture` or `HiddenMarkov`, absorb directly.
2. Else, create `MM` wrapper, call its `fit`, then extract `model.gmm` or `model.hm`.
3. Store under appropriate container.

**Bulk Getters** (`get_means`, `get_variances`, `get_weights`, `score`, `get_log_likelihoods`):

* Accept `data_ids=None` (all), single ID, or list. Return dict or scalar accordingly.

**export\_model(data\_id)**: retrieves the object, not state dict.

**unfit(data\_id)**: remove stored model.

### Cerebrum: Top-Level Container

**Initialization**: empty `ModuleDict`.

**add\_model(model, model\_id)**: insert any `nn.Module` under a key, ensuring uniqueness.

**fit\_and\_add(data, model\_type, ...)**:

* For `gmm`/`hmm`, uses a transient `MMMan` to fit and then registers that manager under `model_id`.
* For `mmm`, directly instantiates `MultiMixtureTransformer`, runs custom training loop combining:

  * Reconstruction (`MSE`) + KL divergence for VAE + negative HMM log‐likelihood.
  * Optional KL annealing schedule.
  * Gradient clipping and logging.

**export\_model(model\_id)** / **import\_model(model\_id)**: wrap `.state_dict()` ops.

**\_select\_data(mm, fn, data\_ids, …)**: utility to dispatch bulk model queries on an MMMan‐style object.

**get\_means**, **get\_variances**, **get\_weights**, **score**, **get\_log\_likelihoods**: top‐level front ends that locate the appropriate manager and call `_select_data`.

**assimilate(other\_path)**:

1. Load another `Cerebrum` from disk.
2. For each submodel, deep copy architecture, export its state to a temp file, import into the copy, and register under a new ID.
3. Clean up temp files. Return updated list of IDs.

**save(path)** / **load(path)**: serialize entire `Cerebrum`.

### MultiMixtureTransformer: Hybrid Sequence Model

**Subcomponents**:

* VAE encoder/decoder (`Encoder`, `Decoder`) for per‐step latent coding.
* RNN‐HMM (`RecurrentNetwork` + `HiddenMarkov`) for state‐space modeling.
* Transformer (`TimeSeriesTransformer`) for sequence‐to‐sequence latent mapping.
* Per‐dimension weights (`pred_`, `recog_`, `gen_`, `reg_weights`) controlling various tasks.
* Optional autoregressive decoder (`auto_decoder` + `to_vocab`) for token‐level tasks.

**Key Methods**:

* `reparameterize(mu, logvar)`: sample via reparametrization trick.
* **Forward**:

  1. If 3‐D input (T,B,D): encode each time step → list of `z_t`, stack into `(T,B,Z)`.
  2. Single‐step input: encode once → `(B,Z)`.
  3. Decode zs → reconstructions, shape matches input.
  4. RNN → emissions, transitions; HMM → log‐likelihood trajectory.
  5. Optional transformer pass if `tgt` provided.
  6. Return dict with all intermediate tensors.
* `loss(x, outputs)`: sum of reconstruction loss (`MSE`), KL divergence, and HMM NLL.
* `training_step(x, optimizer)`: wrapper for one iteration of forward, loss, backprop, weight update (+ auxiliary rule loss).
* **Predict/recognize/generate**:

  * `predict(x)`: one‐step lookahead by reweighting latent dims.
  * `recognize(x, tgt_z)`: cross‐attend latent to target embedding, then decode.
  * `generate(num_steps)`: sample initial state from HMM, roll through transitions + mixture sampling, decode each step.
* **Regression tasks**:

  * `regression(context_sequence)`: builds a “latent plan,” blockwise diffusion refinement, CoRe² drafting of multiple token sequences, scoring via `DraftScorer`, best‐draft refinement, entropy regularization.
  * `regression_loss(...)`: teacher‐forced decoding on ground truth with plan infusion and entropy penalty.

### Supporting Components

#### Encoder and Decoder

Detailed feedforward layers mapping input ↔ latent space, with ReLU activations and sigmoid output constraints.

#### TinyDiffusionBlock

Blockwise MLP‐based denoiser implementing a simple residual step:
$x \leftarrow x + \alpha \cdot f(x)$, $\alpha=0.1$.

#### DraftScorer

Embeds discrete drafts, cross‐attends to a continuous plan under mixed precision, autoregressively decodes to logits, and computes negative log‐likelihood of the original draft.

#### RecurrentNetwork and HiddenMarkov

RNN outputs parameterize state‐emission log‐probs and transition log‐probs; HMM implements the forward algorithm in log‐space for exact sequence likelihoods.

#### GaussianMixture

Vectorized computation of log‐likelihood via Mahalanobis distances, normalization constants, and log‐sum‐exp across components.

#### TimeSeriesTransformer

Standard PyTorch Transformer with learned input/output projections, enabling flexible encoder–decoder over time‐series.

# Data Flow and Computational Graph

Illustrate a detailed diagram: (1) input → encoder → latent zs; (2) zs → decoder → recon; (3) zs → RNN → emissions; (4) emissions + transitions → HMM NLL; (5) zs + tgt → Transformer → autoregressive outputs.

# Training Procedures

## GMM/HMM Maximum Likelihood

Derive gradients w\.r.t. mixture logits, means, log‐variances, and transition matrices using PyTorch automatic differentiation.

## Hybrid Model Training

Detailed schedule:

1. Forward pass through all submodules.
2. Compute reconstruction MSE, clamp `logvar` within \[−10,10] to avoid instability.
3. Compute KLD = $-0.5 ∑(1 + \logσ² − μ² − σ²)$.
4. Clamp HMM log‐likelihood to avoid infinities (e.g. \[−1e6,1e6]).
5. Optionally apply KL annealing: $w_{KL}=\min(1,e/E_{anneal})$.
6. Backpropagate total loss, clip gradients at `clip_norm`, optimizer step.

## Auxiliary Rule Losses

Users can `add_rule(name,label,target,reward)` to enforce scalar‐mean constraints on per‐dim weight vectors via squared‐error penalties.

# Inference and Generation

## Scoring and Sampling

`score(X)` returns mean log‐likelihood; for HMMs, sums over sequences.

## Regression and Autoregressive Generation

Detailed pseudo‐code for `regression()` including block splitting, denoiser instantiation per block, top‑p sampling, cross‑attention loops, and final refinement.

# Performance Considerations

* Memory: storing multiple submodels vs. single large hybrid.
* Computation: per‐step encoding and HMM forward algorithm is $O(TS^2)$. Transformer adds $O(T^2 d)$.
* Mixed precision: Transformer cross‑attention under `torch.amp.autocast` on GPU.

# Example Usage Patterns

Below are comprehensive examples covering all major workflows.

## 1. Basic GMM/HMM Operations

```python
from cerebrum import Cerebrum, MMMan

brain = Cerebrum()
# Fit a new GMM on data matrix X_gmm
gmm_id = brain.fit_and_add(data=X_gmm, model_type='gmm', n_components=4, model_id='my_gmm')
# Score test data under this GMM
gmm_score = brain.score('my_gmm', X_test)
# Retrieve component parameters
means = brain.get_means('my_gmm')
variances = brain.get_variances('my_gmm')
weights = brain.get_weights('my_gmm')

# Fit an HMM on sequence data
hmm_id = brain.fit_and_add(data=seq_data, model_type='hmm', n_components=3, n_mix=2)
# Per-sequence log-likelihoods
lls = brain.get_log_likelihoods(hmm_id, seq_data)
# Remove an unused model
brain.models['my_gmm'].unfit('my_gmm')  # for MMMan-based managers
```

## 2. Managing Multiple Submodels with MMMan

```python
# Directly use MMMan
mmman = MMMan()
# Absorb pre-trained GMM instance
from cerebrum import GaussianMixture
pre_gmm = GaussianMixture(5, 10)
mmman.fit(data=pre_gmm, model_type='gmm', data_id=0)
# Fit HMM and GMM together
mmman.fit(data=lh_seq, model_type='hmm', n_components=2, n_mix=3, data_id='hmm_seq')

# Bulk queries
all_means = mmman.get_means()  # dict of all stored
single_means = mmman.get_means(0)  # numpy array
scores = mmman.score(seq_data_batch)  # dict of scores per model
```

## 3. Hybrid MMM: MultiMixtureTransformer

```python
# Fit-and-add an end-to-end hybrid model
mmm_id = brain.fit_and_add(
    data=ts_data,
    model_type='mmm',
    input_dim=16,
    hidden_dim=64,
    z_dim=32,
    rnn_hidden=128,
    num_states=4,
    n_mix=2,
    trans_d_model=32,
    trans_nhead=4,
    trans_layers=3,
    output_dim=16,
    epochs=200,
    kl_anneal_epochs=50,
    clip_norm=2.0,
    weight_decay=1e-4
)
# One-step prediction
x_next = brain.models[mmm_id].predict(x_current)
# Recognition toward a target latent z
z_target = torch.randn_like(x_current)  # example
reconstructed = brain.models[mmm_id].recognize(x_current, tgt_z=z_target)
# Full sequence generation
gen_seq = brain.models[mmm_id].generate(num_steps=10, batch_size=2)

# Regression / CoRe² drafting
logits, entropy_penalty = brain.models[mmm_id].regression(context_sequence)
loss = brain.models[mmm_id].regression_loss(context_sequence, target_tokens)
```

## 4. Auxiliary Features & Meta-Operations

```python
# Rule-based auxiliary loss
model = brain.models[mmm_id]
model.add_rule(name='pred_balance', label='pred', reward_target=0.5, reward=10.0)
# During training_step, back_loss incorporates these rules
optimizer = torch.optim.Adam(model.parameters())
loss = model.training_step(x_batch, optimizer)

# Export & Import state
state = brain.export_model(mmm_id)
brain.import_model(mmm_id, state)

# Check stored IDs
ids_map = brain.models['MMMan'].check_data()  # or brain.check_data()

# Assimilate another Cerebrum
brain.assimilate('other_brain.pt')

# Complete save/load
brain.save('cerebrum_full.pt')
new_brain = Cerebrum.load('cerebrum_full.pt')
```

# Component Summary Table

| Component               | Role                                            | Key Methods / Complexity                       |
| ----------------------- | ----------------------------------------------- | ---------------------------------------------- |
| GaussianMixture         | Unconditional mixture                           | `log_prob`: O(NKD), vectorized                 |
| HiddenMarkov            | HMM with GMM emissions                          | `log_prob`: O(TS^2 + TSMD)                     |
| MM                      | Single‐model wrapper                            | `.fit`: gradient MLE, `.score`: mean log\_prob |
| MMMan                   | Manager of multiple MM instances                | Bulk getters across IDs                        |
| TimeSeriesTransformer   | Seq2Seq Transformer                             | O(T^2 d + TBd)                                 |
| TinyDiffusionBlock      | Blockwise denoiser                              | O(BD^2) per block                              |
| DraftScorer             | Draft scoring via cross‑attention & NLL         | O(T^2 d + BV d)                                |
| MultiMixtureTransformer | Hybrid end‐to‐end VAE+HMM+diffusion+Transformer | Aggregates all above, configurable components  |
| Cerebrum                | Top‐level container + assimilation              | `assimilate`: file I/O + model deep copy       |

# Conclusion

Cerebrum provides a scalable, extensible scaffolding for hybrid probabilistic‐neural models.&#x20;

# Licensing

**Unified Source-Available License and Contributor Agreement**
© 2025 Chance Brownfield
Last updated: June 24, 2025

1. **DEFINITIONS**

   1. **Licensed Work**: any code, model architecture specifications, documentation, and pretrained model weights published by the Licensor.
   2. **Pretrained Model**: any model weights publicly released by the Licensor, trained on the Licensed Work.
   3. **Contribution**: any submission (code, documentation, pretrained models, or other materials) made by a third party to this project.
   4. **Derivative Work**: has the meaning given under applicable copyright law.

2. **PRETRAINED MODELS: CC BY-NC-ND TERMS**

   1. All Pretrained Models are licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
   2. You may copy and redistribute the Pretrained Models **only** for non-commercial purposes, provided you give appropriate attribution:
      “Pretrained model by Chance Brownfield, licensed CC BY-NC-ND 4.0.”
   3. You may **not** create derivative works of the Pretrained Models or use them commercially without express written permission from Chance Brownfield.

3. **CODE & ARCHITECTURE: BUSINESS SOURCE LICENSE 1.1**

   1. You may copy, modify, create Derivative Works of, and use the Licensed Work **only** for non-production and non-commercial purposes, without further permission.
   2. All production or commercial use—including internal deployment, offering services to third parties, retraining, inference for payment, or embedding in products—**requires explicit written permission** and/or a commercial license from Chance Brownfield.
   3. You may **not** train, refine, or develop any new model using this architecture without the express authorization of Chance Brownfield.
   4. **Change Date:** On June 24, 2029, this section (3) automatically converts to the terms of the GNU General Public License version 2.0 or later, and the restrictions in 3.1–3.3 no longer apply.
   5. Rights granted under this section terminate automatically upon any breach of these terms. Continued use after termination requires a commercial license or cessation of use.

4. **CONTRIBUTOR LICENSE AGREEMENT (CLA)**
   By submitting a Contribution to this project, you agree that:

   1. You own or have the rights to your Contribution.
   2. You grant the Licensor a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to reproduce, prepare Derivative Works of, distribute, and publicly display or perform your Contribution, and to sublicense these rights under any terms (including proprietary or commercial licenses).
   3. You grant the Licensor a patent license to use any patents you hold that would necessarily be infringed by your Contribution.
   4. You waive all moral rights and any claims against the Licensor arising from use of your Contribution.
   5. You acknowledge and support the Licensor’s enforcement of Sections 2 and 3: all commercial use, retraining, deployment, or derivative architecture development requires explicit permission from Chance Brownfield.

5. **WARRANTY & LIABILITY**
   The Licensed Work and any Contributions are provided “AS IS”, without warranty of any kind. The Licensor and contributors are not liable for any claims or damages arising from use.

6. **GOVERNING LAW**
   This entire Agreement is governed by the laws of the USA, without regard to conflict-of-law principles.

7. **CONTACT & COMMERCIAL LICENSING**
   For any commercial licensing inquiries, or to request permission for training or derivative work, please contact:
   Chance Brownfield
   [ChanceBrownfield3515@gmail.com](mailto:ChanceBrownfield3515@gmail.com)
