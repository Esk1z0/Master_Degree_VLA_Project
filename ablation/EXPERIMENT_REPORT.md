# Action Expert Layer Ablation Study: Technical Report

This document details the experimental design, evaluation protocols, raw execution logs, and empirical results for the systematic **Layer Ablation Study** (*Action Expert Layercut*) conducted on the SmolVLA Vision-Language-Action model.

---

## 1. Experiment Design & Execution

### Objective & Theoretical Motivation
The Action Expert of SmolVLA consists of 16 transformer layers responsible for processing visual-linguistic tokens and autoregressively decoding low-level joint trajectory actions. To incorporate new multimodal sensory modalities (such as stereo depth point clouds in SmolVLA-D and visual memory in SmolVLA-MD) without degrading the pre-trained action distribution, it is crucial to determine where the network exhibits natural tolerance to perturbations versus where it possesses critical structural bottlenecks.

```
SmolVLA Action Expert (16 Transformer Layers):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ L0    │ L1    │ L2    │ L3    │ L4    │ L5    │ L6    │ L7    │ L8    │ L9    │ L10   │ L11   │ L12   │ L13   │ L14   │ L15   │
├───────┼───────┴───────┴───────┼───────┼───────┼───────┴───────┴───────┼───────┼───────┼───────┼───────┼───────┴───────┼───────┤
│ Entry │   CRITICAL (1-3)      │ Inter │ Crit  │ TOLERANT WINDOW (6-8) │ Trans │ Trans │ Inter │ Trans │ CRITICAL(13-14│ Exit  │
└───────┴───────────────────────┴───────┴───────┴───────────────────────┴───────┴───────┴───────┴───────┴───────────────┴───────┘
                                                ▲ Target Residual Injection Window (SmolVLA-D)
```

### Ablation Mechanism (`smolvla_layercut`)
- **Skip-Block Identity Passthrough:** Rather than setting weights to zero or pruning modules (which drastically shifts intermediate activation statistics), ablated layers are patched to `None` in `get_model_layers`. The activation tensor passes through the layer untouched via pure residual identity connections without evaluating multi-head attention or MLP blocks.
- **Prefix-Free Denoising Patch:** In `forward_attn_layer`, handling was introduced to safely bypass empty tensor lists when `inputs_embeds=[None, suffix]` during diffusion/flow-matching denoising iterations.
- **Model Checkpoint:** All physical trials utilized a fine-tuned manipulation policy checkpoint (`outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model`), trained on the unified physical dataset (`Esk1z0/tfm_layer_ablation_batch_1` + `batch_2`).
- **Sanity Verification:** Every configuration was pre-verified with [`verify_layercut.py`](verify_layercut.py) to ensure exact layer masking before robot deployment.

### Experimental Phases & Configurations (21 Models, 315 Physical Trials)
The study was organized across three complementary tiers:

1. **Phase 0 — Numerical Offline Divergence Sweep (`xai_ablation.py`):**
   - Evaluated 135 contiguous layer window combinations (window sizes 1 to 15, across all valid layer offsets) on a synthetic multi-modal batch.
   - Measured output action divergence (MSE and MAE) against the unablated baseline to map gross sensitivity landscapes prior to physical deployment.
2. **Phase 1 — Single-Layer Physical Ablation:**
   - 17 physical model configurations: the unablated `base` policy plus 16 individual single-layer ablations (`layercut_0` to `layercut_15`).
3. **Phase 2 — Multi-Layer Contiguous Range Ablation:**
   - 4 multi-layer configurations (`layercut_range_6_7`, `layercut_range_6_7_8`, `layercut_range_7_8_9`, `layercut_range_9_10_11`) targeting the identified tolerant plateau and its boundary to test if single-layer tolerance is additive.

### Hardware Setup & Benchmark Scenes
- **Robot Hardware:** 6-DoF **SO-101 robotic arm** equipped with standard parallel jaw gripper, top RGB context camera, and wrist-mounted SVPRO dual-lens stereo camera (2560x800).
- **Physical Evaluation Design:** Evaluated across **15 physical trials per model** (14 trials for `layercut_6` and `layercut_13` due to single-trial data dropouts; total 315 physical executions).
- **Controlled Layouts:** Trials were split into 3 standardized physical configurations (5 trials each):
  - `training`: Standard object placement configurations present in the training distribution.
  - `no_visto` (*unseen*): Unseen spatial positions testing spatial generalization.
  - `stress`: Tightly spaced or rotated object configurations testing precision manipulation.
- Four physical objects on a 3x3 table grid (`A1` to `C3`): Black Star (`starBlack`), Orange Star (`starOrange`), Black Cube (`cubeBlack`), and Orange Cube (`cubeOrange`).

---

## 2. Evaluation Protocol & Scoring Rubric

The benchmark task requires sorting all 4 objects: picking the 2 stars and placing them inside a container receptacle, and picking the 2 cubes and positioning them onto delimited planar target marks.

### 9 Binary Evaluation Criteria
Each physical trial is evaluated across 9 strictly defined binary criteria (1 = success, 0 = failure):

```
                                 9-Point Binary Rubric
 ┌───────────────────────────────────────┬───────────────────────────────────────┐
 │          Star Sorting (4 pts)         │          Cube Placement (4 pts)       │
 ├───────────────────────────────────────┼───────────────────────────────────────┤
 │ 1. Black Star Grasp & Lift            │ 5. Black Cube Grasp & Lift            │
 │ 2. Black Star Receptacle Placement    │ 6. Black Cube Target Mark Placement   │
 │ 3. Orange Star Grasp & Lift           │ 7. Orange Cube Grasp & Lift           │
 │ 4. Orange Star Receptacle Placement   │ 8. Orange Cube Target Mark Placement  │
 └───────────────────────────────────────┴───────────────────────────────────────┘
                                     │
                   9. Clean Episode Completion (1 pt)
```

### Detailed Criteria Definitions

#### 1. Grasp and Lift (`*_agarre`)
- **Score 1:** The gripper achieves firm closure on the target object, completely breaks contact with the table surface, and maintains stability through the initial trajectory.
- **Score 0:** The robot merely pushes/drags the object without lifting, drops the object immediately after lifting, or pinches an object against another obstacle without true grasp.

#### 2. Star Placement (`estrella_*_destino`)
- **Score 1:** The star is released inside the receptacle boundaries and remains inside at the end of the episode without being knocked out by subsequent arm motions.
- **Score 0:** The star rests on the rim, falls outside, or is knocked out during subsequent actions.

#### 3. Cube Placement (`cubo_*_destino`)
- **Score 1:** The cube is released onto the target mark with its base completely inside the designated boundary zone.
- **Score 0:** The cube falls outside the target zone, partially overhangs the boundary mark, or is displaced outside during subsequent actions.

#### 4. Clean Completion (`finalizacion_limpia`)
- **Score 1:** The robot finishes the full sorting sequence within the allotted episode time limit without entering kinematic limit locks, without erratic trajectory oscillations, and without requiring emergency operator intervention.
- **Score 0:** The robot freezes, oscillates endlessly, crashes violently into the table/boundary, or requires physical manual intervention.

### Aggregated Metrics
- **Normalized Score:** $\text{Score} = \frac{\text{Raw Points}}{9} \in [0.0, 1.0]$.
- **Complete Task Success Rate:** $\text{Success} = 1 \iff \text{Raw Points} = 9$ (100% of subtasks achieved cleanly).
- **Failure Taxonomy:** Each failure is categorized by mode (`modo_fallo`: `fallo_agarre`, `colision`, `bloqueo`, `desorden`, `otro`), execution phase (`fase_fallo`), and target object (`objeto_fallo`).

---

## 3. Results & Key Findings

### Master Experimental Results Table (All 21 Model Configurations)

| Model Variant | N | Mean Score $\mu$ | Std $\sigma$ | Complete Success % | Grasp Star N | Grasp Star O | Grasp Cube N | Grasp Cube O | Place Star N | Place Star O | Place Cube N | Place Cube O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`base` (SmolVLA)** | 15 | **0.830** | 0.234 | **60.0%** | 0.867 | 0.933 | 0.800 | 0.867 | 0.867 | 0.933 | 0.800 | 0.800 |
| `layercut_0` | 15 | 0.645 | 0.310 | 26.7% | 0.667 | 0.800 | 0.667 | 0.667 | 0.667 | 0.733 | 0.667 | 0.667 |
| `layercut_1` | 15 | 0.176 | 0.091 | 0.0% | 0.067 | 0.400 | 0.333 | 0.000 | 0.067 | 0.400 | 0.333 | 0.000 |
| `layercut_2` | 15 | 0.235 | 0.196 | 0.0% | 0.333 | 0.333 | 0.267 | 0.133 | 0.333 | 0.333 | 0.267 | 0.133 |
| `layercut_3` | 15 | 0.257 | 0.142 | 0.0% | 0.467 | 0.333 | 0.333 | 0.067 | 0.467 | 0.267 | 0.333 | 0.067 |
| `layercut_4` | 15 | 0.607 | 0.383 | 40.0% | 0.600 | 0.800 | 0.533 | 0.600 | 0.600 | 0.800 | 0.533 | 0.600 |
| `layercut_5` | 15 | 0.191 | 0.183 | 0.0% | 0.400 | 0.267 | 0.200 | 0.000 | 0.400 | 0.267 | 0.200 | 0.000 |
| **`layercut_6`** | 14 | **0.613** | 0.247 | 14.3% | 0.714 | 0.786 | 0.429 | 0.714 | 0.714 | 0.786 | 0.429 | 0.714 |
| **`layercut_7`** | 15 | **0.585** | 0.272 | 20.0% | 0.667 | 0.733 | 0.600 | 0.533 | 0.667 | 0.733 | 0.600 | 0.533 |
| **`layercut_8`** | 15 | **0.533** | 0.319 | 20.0% | 0.600 | 0.800 | 0.467 | 0.467 | 0.600 | 0.733 | 0.467 | 0.467 |
| `layercut_9` | 15 | 0.473 | 0.300 | 13.3% | 0.467 | 0.733 | 0.533 | 0.333 | 0.467 | 0.733 | 0.533 | 0.333 |
| `layercut_10` | 15 | 0.429 | 0.349 | 13.3% | 0.333 | 0.533 | 0.467 | 0.533 | 0.333 | 0.533 | 0.467 | 0.533 |
| `layercut_11` | 15 | 0.541 | 0.342 | 20.0% | 0.533 | 0.800 | 0.533 | 0.467 | 0.533 | 0.800 | 0.533 | 0.467 |
| `layercut_12` | 15 | 0.355 | 0.250 | 0.0% | 0.533 | 0.733 | 0.200 | 0.133 | 0.533 | 0.733 | 0.200 | 0.133 |
| `layercut_13` | 14 | 0.251 | 0.146 | 0.0% | 0.429 | 0.267 | 0.133 | 0.267 | 0.400 | 0.267 | 0.133 | 0.267 |
| `layercut_14` | 15 | 0.169 | 0.171 | 0.0% | 0.067 | 0.267 | 0.133 | 0.333 | 0.067 | 0.267 | 0.133 | 0.267 |
| `layercut_15` | 15 | 0.369 | 0.173 | 0.0% | 0.800 | 0.600 | 0.200 | 0.533 | 0.467 | 0.533 | 0.000 | 0.200 |
| `layercut_range_6_7` | 15 | 0.383 | 0.284 | 13.3% | 0.267 | 0.667 | 0.467 | 0.267 | 0.267 | 0.667 | 0.467 | 0.267 |
| `layercut_range_6_7_8` | 15 | 0.354 | 0.212 | 0.0% | 0.333 | 0.467 | 0.467 | 0.333 | 0.333 | 0.467 | 0.467 | 0.333 |
| `layercut_range_7_8_9` | 15 | 0.198 | 0.104 | 0.0% | 0.267 | 0.333 | 0.333 | 0.000 | 0.200 | 0.333 | 0.333 | 0.000 |
| `layercut_range_9_10_11` | 15 | **0.125** | 0.097 | 0.0% | 0.200 | 0.200 | 0.267 | 0.000 | 0.200 | 0.133 | 0.133 | 0.000 |

### Key Empirical Takeaways

1. **Critical Structural Bottlenecks:**
   - **Early Layers (1, 2, 3, 5):** Disabling layers 1–3 drops performance drastically ($\le 0.257$ mean score, $0\%$ success rate). These layers build fundamental spatial token bindings.
   - **Late Layers (13, 14):** Disabling layer 14 produces the single worst individual ablation score ($0.169$). These layers are responsible for high-precision end-effector trajectory synthesis.
2. **Contiguous Tolerant Plateau (Layers 6–8):**
   - Layers 6, 7, and 8 show the highest retention of competence among all intermediate layers ($\mu = 0.613, 0.585, 0.533$ with success rates of $14.3\%–20.0\%$).
   - While `layercut_0` ($0.645$) and `layercut_4` ($0.607$) score well, they are isolated points with steep adjacent drops.
3. **Non-Additivity of Tolerance:**
   - Disabling multiple layers simultaneously reveals that tolerance is non-linear. Disabling the full 6–8 block (`range_6_7_8`) drops performance to $0.354$, while shifting the window towards later layers (`range_9_10_11`) causes catastrophic breakdown ($\mu = 0.125$, $0\%$ success).
4. **Architectural Choice for SmolVLA-D:**
   - The contiguous tolerant window **layers 6, 7, and 8** was selected for 3D depth injection (`depth_injection_layers=[6, 7, 8]`).
   - Rather than replacing or pruning, depth features are injected via **zero-initialized residual cross-attention adapters**, preserving baseline motor capabilities while progressively exposing 3D geometric information.

---

## 4. Code & Artifact Links

- [`resultados_ablacion.ipynb`](resultados_ablacion.ipynb): Full analysis notebook with statistical tests and distributions.
- [`prepare_layercut_checkpoint.py`](prepare_layercut_checkpoint.py): Utility script to configure layercut checkpoints.
- [`verify_layercut.py`](verify_layercut.py): Model verification utility.
- [`evaluations/`](evaluations/): Directory containing raw CSV logs for all 21 models.
- [`graphs/`](graphs/): Visualizations and plotting generation scripts.
- [`EXPERIMENT_REPORT_ES.md`](EXPERIMENT_REPORT_ES.md): Spanish version of this technical report.
