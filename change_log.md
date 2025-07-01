**Changelog for Cerebrum — 2025-06-28**

![Version](https://img.shields.io/badge/version-0.1.1-red)


**Added**
* **New forward method for Cerebrum class**
* **Dynamic Expert Assurance**

  * Automatically spawns an initial adapter expert in the `forward` pass when no experts exist.
  * Ensures seamless integration of new tasks without manual expert initialization.

* **Confidence & Entropy Checks**

  * Computes mean confidence (`max` of gating probabilities) and entropy per batch.
  * Introduces thresholds (`conf_thresh`, `ent_thresh`) to trigger on-the-fly expert spawning.

* **Re-routing After Growth**

  * Recomputes gating logits and probabilities immediately after spawning new experts.
  * Guarantees updated routing decisions in the same forward pass.

* **MoE Mixing Pipeline**

  * Unified legacy and adapter experts under a single MoE mixing loop.
  * Temperature-scaled expert outputs with per-key accumulation in one unified output dict.

* **Load-balancing Loss**

  * Added entropy-based penalty term (`load_loss`) on the average expert usage distribution.
  * Encourages balanced utilization of all active experts.

* **Debug Information**

  * Expanded output dictionary with a `debug_info` sub-dict containing:
  * `confidence`: Scalar confidence value.
  * `entropy`: Scalar entropy value.
  * `n_experts`: Current count of active experts.

* **Logit Masking Logic**

  * Implemented masking to ensure logits for inactive expert slots are set to
    `-inf` only when `current_experts < max_experts`.

* **Stability in Softmax**

  * Added numerical safeguard (`1e-9`) in entropy computation to prevent `log(0)` errors.

---