# Physical Hardware Final Evaluation: Technical Report

This document details the experimental design, evaluation protocols, raw execution logs, and empirical results for the physical hardware evaluation comparing the four SmolVLA model variants on the **SO-101** robotic arm.

---

## 1. Experiment Design & Execution

### Objective & Scope
The final hardware evaluation is an end-to-end **behavioral validation benchmark** designed to evaluate whether the multimodal capabilities introduced in SmolVLA-M (temporal visual memory) and SmolVLA-D (3D stereo depth injection) translate into measurable real-world manipulation improvements, spatial accuracy, and distractor resilience on low-cost physical hardware.

```
Evaluated Model Variants (All Checkpoints v2):
┌────────────────────┬─────────────────────────────────────────────────┬──────────────────────┐
│ Model Variant      │ Architecture & Sensory Additions                │ Checkpoint Status    │
├────────────────────┼─────────────────────────────────────────────────┼──────────────────────┤
│ 1. SmolVLA-Vanilla │ Monocular/Stereo RGB baseline                   │ 22,500 st. (100.0%)  │
│ 2. SmolVLA-M       │ Causal Temporal Visual Memory (K=6 frames)      │ 24,000 st. (~53.3%)  │
│ 3. SmolVLA-D       │ 3D Depth Injection in Layers 6-8 (Cross-Attn)   │ 22,500 st. (100.0%)  │
│ 4. SmolVLA-MD      │ Combined Multimodal (Depth + Temporal Memory)   │ 24,000 st. (~53.3%)  │
└────────────────────┴─────────────────────────────────────────────────┴──────────────────────┘
```

### Live Inference & Depth Pipeline
- **Hardware Platform:** 6-DoF **SO-101 robotic arm** operated with LeRobot. Dual camera setup: fixed overhead context camera (`camera1`) and wrist-mounted stereo camera (`camera2`, 2560x800).
- **Live 3D Point Cloud Injection:** Evaluated in real-time using `lerobot-record` with `--depth_calib_path camera/calib_stereo.npz`. OpenCV SGBM+WLS disparities are computed on each control tick, projecting calibrated 3D geometric point clouds into the policy's residual adapters.

### Benchmark Scenario Design (12 Physical Episodes per Model)
To reliably isolate memory and depth effects without confounding variance, all 4 models were evaluated across **12 standardized physical trials** (6 without photograph distractor, 6 with photograph distractor), executed in block:

```
Physical Table Layout (Grid A1–C3):
┌──────────────┬──────────────┬──────────────┐
│      A1      │      A2      │      A3      │
│   [Empty]    │ Orange Star  │  Black Cube  │
├──────────────┼──────────────┼──────────────┤
│      B1      │      B2      │      B3      │
│ [Receptacle] │  Black Star  │ *Orange Cube*│ ◄ Target Occluded Object
├──────────────┼──────────────┼──────────────┤
│      C1      │      C2      │      C3      │
│ [Photo Dist.]│   [Empty]    │ [Target Zone]│
└──────────────┴──────────────┴──────────────┘
```

1. **`combinado_sin_foto` (Episodes `C-01` to `C-06`):**
   - Focus: Isolates temporal memory.
   - Setup: Real orange cube in position `B3` becomes dynamically occluded by the robot's arm while picking central stars (`B2`/`A2`). No photograph distractor present.
2. **`combinado_con_foto` (Episodes `C-07` to `C-12`):**
   - Focus: Maximum stress test (Depth + Memory).
   - Setup: Identical baseline layout, with a high-fidelity 2D printed photograph of the orange cube placed in cell `C1`.

---

## 2. Evaluation Protocol & Extended Scoring Rubric

Because episodes with photo distractors evaluate additional perceptual capabilities, the scoring rubric is divided into distinct modular blocks:

### Scoring Structure (11 pts Max without Photo / 13 pts Max with Photo)

```
                            Modular Evaluation Rubric
 ┌──────────────────────────────────────────┬──────────────────────────────────────────┐
 │        Block B: Dexterity (9 pts)        │         Block C: Memory (2 pts)          │
 ├──────────────────────────────────────────┼──────────────────────────────────────────┤
 │ • 4 Grasp Criteria (Stars & Cubes)       │ • recuerda_objeto_ocluido (1/0)          │
 │ • 4 Placement Criteria (Box & Mark)      │ • completa_objeto_ocluido (1/0)          │
 │ • finalizacion_limpia (1/0)              │                                          │
 └──────────────────────────────────────────┴──────────────────────────────────────────┘
                                     │
 ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
 │      Block D: 3D / Photo Distractor (2 pts, evaluated only in C-07 to C-12)         │
 ├─────────────────────────────────────────────────────────────────────────────────────┤
 │ • selecciona_real_antes_foto (1/0)   │ • evita_intento_agarre_foto (1/0)            │
 │ • intentos_agarre_foto (0/1/2 count) │                                              │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

### Criteria Definitions & Validation Gates

#### 1. Experimental Quality Gates (Block A)
- `setup_validado` (1/0): Confirms exact physical object grid placement.
- `objeto_visible_inicio` (1/0): Target occluded object was visible before trajectory start.
- `oclusion_verificada` (1/0): The robotic arm physically occluded the target object during motion.
- `registro_valido` (1/0): Gate indicator ($1$ only if setup, visibility, and occlusion are all verified).

#### 2. Dexterity & Task Execution (Block B — 9 pts)
- Grasp and placement criteria for all 4 objects following the standardized binary rubric.
- `finalizacion_limpia` (1/0): Unassisted completion without kinematic lock, infinite oscillations, or crashes.

#### 3. Temporal Memory (Block C — 2 pts, evaluated across all 12 episodes)
- `recuerda_objeto_ocluido` (1/0): Following occlusion, the robot smoothly re-directs its gaze and trajectory back toward the occluded object.
- `completa_objeto_ocluido` (1/0): The robot successfully completes manipulation and placement of the occluded object.

#### 4. 3D Geometric Discrimination (Block D — 2 pts, evaluated in episodes `C-07` to `C-12`)
- `selecciona_real_antes_foto` (1/0): The robot's first grasp attempt targets the physical 3D object rather than the 2D photograph distractor.
- `evita_intento_agarre_foto` (1/0): The gripper never closes on or collides with the photograph distractor during the entire episode.
- `intentos_agarre_foto` (Scale 0/1/2): Categorical tracking ($0$ = gaze only, $1$ = single grasp attempt on photo, $2$ = multiple grasp attempts on photo).

#### 5. Safety & Failure Flags (Block E)
- `timeout` (1/0): Episode exceeded allocated time limit ($120$ s) or looped indefinitely.
- `intervencion_emergencia` (1/0): Physical emergency stop by operator due to erratic oscillations or dangerous collisions.

### Score Normalization
$$\text{Score}_{\text{sin\_foto}} = \frac{\text{Raw Points}}{11} \in [0.0, 1.0], \quad \text{Score}_{\text{con\_foto}} = \frac{\text{Raw Points}}{13} \in [0.0, 1.0]$$
- **Complete Success:** $\text{Success} = 1 \iff \text{registro\_valido} = 1 \land \text{Score} = 1.0 \land \text{timeout} = 0 \land \text{intervencion} = 0$.

---

## 3. Results & Cross-Model Behavioral Analysis

### Master Performance Summary

| Model Variant | Global Normalized Score | Without Photo Score | With Photo Score | Complete Success Rate | Timeouts | Emergency Stops |
|---|---|---|---|---|---|---|
| **SmolVLA-Vanilla** | 0.356 | 0.545 | 0.167 | 2 / 12 (16.7%) | 10 / 12 | 0 / 12 |
| **SmolVLA-M** | 0.398 | 0.591 | 0.205 | 1 / 12 (8.3%) | 11 / 12 | 0 / 12 |
| **SmolVLA-D** | **0.490** | **0.864** | 0.115 | **3 / 12 (25.0%)** | 9 / 12 | 0 / 12 |
| **SmolVLA-MD** | 0.057 | 0.076 | 0.038 | 0 / 12 (0.0%) | 2 / 12 | **10 / 12** |

---

### In-Depth Model Behavioral Profiles

```
Performance Breakdown by Condition:
  1.0 ┌─────────────────────────────────────────────────────────────┐
      │                                    [0.864]                  │
  0.8 │                                    ██████                   │
      │                  [0.545]  [0.591]  ██████                   │
  0.6 │                  ██████   ██████   ██████                   │
      │                  ██████   ██████   ██████                   │
  0.4 │                  ██████   ██████   ██████                   │
      │                  ██████   ██████   ██████                   │
  0.2 │ [0.167] [0.205]  ██████   ██████   ██████  [0.115]  [0.076] │
      │  ░░░░░   ░░░░░   ██████   ██████   ██████   ░░░░░    ██████ │
  0.0 └─────────────────────────────────────────────────────────────┘
         Vanilla           SmolVLA-M        SmolVLA-D      SmolVLA-MD
        (With / No Photo) (With / No Photo)(With / No Photo)(With / No Photo)
```

1. **SmolVLA-D (Stereo Depth Injection — Top Performer):**
   - Achieves a peak score of **0.864** in clean environments (**+58.5% improvement over baseline Vanilla**).
   - Produced 3 perfect runs out of 6 in `combinado_sin_foto`.
   - Highest occluded memory recall rate ($58.3\%$ recall, $50.0\%$ completion).
   - Exhibits bimodal behavior: excellent manipulation under clean 3D conditions, but vulnerable to 2D photograph planar ambiguity (timeouts on all 6 photo episodes).
2. **SmolVLA-M (Temporal Visual Memory — High Stability):**
   - Most consistent policy: never scored zero in any trial; successfully initiated black star manipulation in all 12 episodes.
   - Occluded recall was high without photo ($83.3\%$), but collapsed when the photo distractor competed for visual attention ($0.0\%$).
3. **SmolVLA-Vanilla (Baseline Policy):**
   - Intermediate performance ($0.356$ global, $0.545$ without photo, 2 complete successes).
   - Complete failure on photo discrimination (`selecciona_real_antes_foto = 0/6`).
4. **SmolVLA-MD (Combined Multimodal — Instability Collapse):**
   - Severe behavioral degradation ($0.057$ global score, $0$ successes, 10 out of 12 episodes terminated by emergency safety stop).
   - Manifested high-frequency motor oscillations ("jitter", kinematic freezes).
   - Analysis indicates that simultaneous cross-attention depth injection and causal recurrent memory hidden states created conflicting gradients during the incomplete training schedule.

### Cross-Cutting Research Findings
- **2D Photo Distractor Floor Effect:** All models failed initial discrimination (`selecciona_real_antes_foto = 0/6`). Planar 2D distractors act as strong visual attractors for standard VLA backbones.
- **Stereo Depth Value:** 3D point cloud injection provides massive precision gains for physical contact tasks and occluded object retrieval when no adversarial 2D distractors are present.
- **Memory-Distractor Interference:** Strong visual distractors actively disrupt the hidden state trajectory of temporal visual memory modules.

---

## 4. Directory Resources

- [`calcular_resultados_evaluacion.py`](calcular_resultados_evaluacion.py): Script to compute metrics and aggregate summary tables from CSV logs.
- [`analisis_significancia.py`](analisis_significancia.py): Statistical significance evaluation script across models.
- [`evaluations/`](evaluations/): Raw execution CSV logs and summaries for each evaluated model variant.
- [`graphs/`](graphs/): Performance comparison figures and script sources.
- [`EXPERIMENT_REPORT_ES.md`](EXPERIMENT_REPORT_ES.md): Spanish version of this technical report.
