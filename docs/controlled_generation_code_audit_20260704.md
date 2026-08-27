# Controlled Generation Code Audit - 2026-07-04

## Scope

This note audits the current GLIGEN + Visual Genome scene-graph branch from code and diagnostic evidence. It is not a formal VG metric report. The goal is to identify why controllable generation remains weak and what to change next.

## Current Evidence

### Diagnostic Stress Subset

The stress subset was selected from the fixed SG2I/VG test split:

```text
H5: /root/autodl-tmp/fixed_split_work/datasets/vg/test.h5
Vocab: /root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json
Sample indices: 6,14,15,23,77,87,144,177,28,42,53,4,8,9,13,56,59,174
Groups: support, containment, vertical, depth, interaction
```

The diagnostic outputs are local at:

```text
artifacts/stress_style_gate_on_off_20260704/
```

### Graph On/Off Probe

The probe used:

```text
Checkpoint: OUTPUT_STANDARD_SG2IM_COMPACT_STYLE_GATE/vg_standard_sg2im_compact_style_gate_1k_20260627/tag00/checkpoint_latest.pth
Sampler: DDIM
Steps: 50
Guidance: 3.0
Seed: 20260704
Samples: 18 fixed stress samples
```

Graph-on and graph-off were identical except:

```text
graph_on: GRAPH_GATE_OVERRIDE=None
graph_off: GRAPH_GATE_OVERRIDE=0.0
```

Pixel difference summary:

```text
num_pairs: 18
mean_mad: 0.1299 / 255
median_mad: 0.1298 / 255
max_mad: 0.4349 / 255
min_mad: 0.0 / 255
```

Interpretation: the graph branch has almost no visible effect on the generated image under the current compact-style checkpoint.

### Relation Token v1/v2 Probe

To test whether explicit relation-as-grounding tokens can produce a stronger but safer control signal, we compared:

```text
graph_on: object grounding + current graph branch
relation_v1: graph_on + up to 5 relation union-box tokens
relation_v2: graph_on + filtered, deduplicated, capped relation union-box tokens
```

The v2 probe used:

```text
Config: configs/vg_standard_sg2im_scene_graph_relation_tokens_v2.yaml
Script: run_standard_sg2im_relation_tokens_v2_stress_eval.sh
Checkpoint: OUTPUT_STANDARD_SG2IM_COMPACT_STYLE_GATE/vg_standard_sg2im_compact_style_gate_1k_20260627/tag00/checkpoint_latest.pth
Sampler: DDIM
Steps: 50
Guidance: 3.0
Seed: 20260704
Samples: same 18 fixed stress samples
Max relation tokens: 3
Predicate filter: spatial/action predicates only
Deduplicate relation tokens: true
```

Metadata check:

```text
relation_v1 relation-token count: min 0, median 2, max 5
relation_v1 duplicate cases: 3 / 18
relation_v2 relation-token count: min 0, median 1, max 3
relation_v2 duplicate cases: 0 / 18
```

Pixel difference summaries:

```text
graph_on vs graph_off:
  mean_mad: 0.1299 / 255
  median_mad: 0.1298 / 255

relation_v1 vs graph_on:
  mean_mad: 10.9884 / 255
  median_mad: 10.1728 / 255
  max_mad: 24.3620 / 255

relation_v2 vs graph_on:
  mean_mad: 6.8635 / 255
  median_mad: 7.0738 / 255
  max_mad: 15.9198 / 255

relation_v2 vs relation_v1:
  mean_mad: 8.5185 / 255
  median_mad: 7.2865 / 255
```

Interpretation: relation tokens are much more visible than the original graph residual path, but v1 can introduce semantic/style drift because noisy or duplicated relations become extra grounding objects. The v2 filtering, deduplication, and lower token cap reduce this drift while preserving visible control signal. This is still a diagnostic result, not proof that spatial relations are correct in the generated images.

Local diagnostic artifacts:

```text
artifacts/stress_style_gate_on_off_20260704/stress_relation_tokens_v1_v2_graph_grid_first12.png
artifacts/stress_style_gate_on_off_20260704/stress_relation_tokens_v2_vs_graph_on_pixel_diff_summary.json
artifacts/stress_style_gate_on_off_20260704/stress_relation_tokens_v2_vs_v1_pixel_diff_summary.json
```

## Code-Level Root Cause

### 1. Graph Injection Is Too Small

File:

```text
ldm/modules/diffusionmodules/scene_graph_grounding_net.py
```

The current path is:

```python
base_tokens = self.encode_base(...)
graph_delta = self.encode_graph_delta(...)
gate = self.resolve_graph_gate(graph_delta)
graph_contribution = gate * graph_delta
return base_tokens + graph_contribution
```

Current style config:

```yaml
graph_gate_init: -3.0
graph_delta_target_ratio: 0.12
```

Since `sigmoid(-3.0) ~= 0.047`, the effective graph contribution is roughly:

```text
0.047 * 0.12 ~= 0.0056
```

This means the graph signal is about 0.5 percent of the base token norm. The graph-on/off stress probe confirms this: graph-on and graph-off produce nearly identical images.

### 2. Auxiliary Losses Mostly Supervise Tokens, Not Images

File:

```text
trainer.py
```

Most losses recompute:

```python
object_tokens = self.model.position_net(**grounding_input)
```

Then they supervise token similarity, token-box contrast, token-relation contrast, or relation-geo prediction. These losses can make the tokenizer encode more graph information, but they do not force the diffusion U-Net to place objects correctly in image space.

The main affected losses are:

```text
object_align_loss
spatial_consistency_loss
object_box_contrastive_loss
relation_contrastive_loss
relation_geo_prediction_loss
relation_geo_consistency_loss
```

The current failure mode is therefore expected: token-level objectives can improve internal representations while generated images remain weakly controlled.

### 3. Relation Geometry Prediction Is Not a Layout Loss

File:

```text
ldm/modules/diffusionmodules/scene_graph_grounding_net.py
```

The masked geometry prediction path is:

```python
masked_geo_features = zeros_like(relation_geo_features)
graph_delta = encode_graph_delta(... masked_geo_features ...)
return predict_relation_geo(graph_delta, relation_edges, relation_embeddings)
```

This trains the graph branch to recover relation geometry from token context. It does not directly supervise the denoised image, U-Net attention, or object locations.

So a decreasing `relation_geo_prediction_loss` does not prove controllable generation improved.

### 4. The Fuser Remains the Real Bottleneck

The graph branch changes grounding tokens, but the downstream GLIGEN fuser must interpret those changes. The previous gates-only and gates-plus-linear probes suggest:

```text
gates_only: visible but small changes
gates_and_linear: larger changes but duller/grayer images
```

This means crude fuser unfreezing can alter image distribution without reliably improving spatial control.

### 5. Quality Distillation Preserves Quality but Suppresses Control

The style config uses:

```yaml
quality_distillation_loss_weight: 2.0
```

This keeps the student close to a graph-disabled teacher. It helps prevent collapse, but if the only strong image-space objective is "stay like the teacher", then weak graph changes are naturally suppressed.

## What This Means

The current strategy is not useless, but it is misaligned with the controllable-generation objective.

The graph branch is learning a graph-aware token residual. The diffusion model is not being forced to use that residual for spatial layout. The current code path is therefore better described as:

```text
scene-graph-aware token regularization
```

not yet:

```text
scene-graph-controlled image generation
```

## Recommended Next Direction

### Step 1: Keep the Diagnostic Stress Set

Before any future training run, use the fixed stress indices:

```text
6,14,15,23,77,87,144,177,28,42,53,4,8,9,13,56,59,174
```

Every proposed change should first show a visible difference on this set before running long training or full FID/IS.

After inspecting part-only failures such as image id 210, a cleaner diagnostic set was created to reduce data/selection noise:

```text
Clean controllability stress set v1:
sample indices: 1,5,14,18,31,36,44,53,56,59,61,62,75,78,124,160,174,185
image ids: 16,126,308,368,668,743,953,1156,1231,1307,1318,1390,1696,1774,2963,3780,4204,4610
source: artifacts/clean_stress_candidates_20260704/clean_controllability_stress18_v1.json
```

This set was selected from fixed VG test candidates with whole-object relations and reduced part-object dominance. It is still diagnostic only, but it is better suited for judging controllability than samples where the selected graph contains only object parts such as `tire` without `bike`.

Clean18 three-way diagnostic result:

```text
Protocol: fixed VG test split, same 18 clean stress samples, DDIM 50, guidance 3.0, seed 20260704.
Variants: graph_off, graph_on, relation_token_v2.

graph_on vs graph_off:
  mean_mad: 0.1315 / 255
  median_mad: 0.0909 / 255
  max_mad: 0.4539 / 255

relation_token_v2 vs graph_on:
  mean_mad: 10.3119 / 255
  median_mad: 8.8500 / 255
  max_mad: 21.3789 / 255

relation_token_v2 metadata:
  relation token count min/median/max: 1 / 1 / 3
```

Interpretation: the original GAT graph residual remains visually negligible even on cleaner samples. Explicit relation tokens are visible, but the visual change is not yet reliably aligned with spatial correctness. This supports prioritizing image-space or attention-space grounding supervision over simply increasing graph-token residual strength.

### Step 2: Add Relation-As-Grounding-Token

Instead of only injecting relation information through a small residual on object tokens, create explicit relation grounding tokens:

```text
text: "person on skateboard"
box: union(person_box, skateboard_box)
```

This reuses GLIGEN's native object-token + box path and gives the fuser a condition it already understands.

Expected advantage:

```text
relations become direct grounding conditions, not only hidden GAT messages
```

### Step 3: Add Attention-Box Alignment

The next real control loss should supervise U-Net grounding attention, not just token space.

For each object token:

```text
attention mass inside its box should be high
attention mass outside its box should be low
```

For relations:

```text
on: subject attention center above object attention center
inside: subject attention lies within object box
left/right/above/below: attention centers respect relative direction
```

This directly targets the failure mode: objects and relations are not reliably realized in image space.

### Step 4: Use Quality Distillation More Selectively

Keep quality distillation for stability, but do not let it dominate spatial learning. A safer schedule is:

```text
early: stronger distillation, weak graph
middle: lower distillation, stronger attention-box loss
late: balanced distillation and spatial losses
```

### Step 5: Do Not Treat FID/IS as the Control Gate

FID/IS should be reported later for image quality. The gate for this chapter should be:

```text
fixed stress-set graph-on/off difference
attention-box alignment
relation spatial compliance
qualitative controllability grids
```

## Immediate Experiment Gate

Before changing the model again, the next experiment should satisfy:

```text
Same stress indices
Same seed
Same sampler
Graph/control variant produces visible layout or object-placement changes
No large quality collapse
Metadata exists for every image
```

If a variant cannot beat the current graph-on/off MAD scale by a large margin, it should not be trained long.

## Relation-As-Grounding-Token Update

Implemented the first relation-token path after this audit:

```text
Each selected relation adds an extra GLIGEN grounding token.
text = "{subject} {predicate} {object}"
box = union(subject_box, object_box)
```

This is enabled by:

```text
enable_relation_grounding_tokens: true
max_relation_grounding_tokens: 5
```

or at eval time:

```text
ENABLE_RELATION_GROUNDING_TOKENS=1
MAX_RELATION_GROUNDING_TOKENS=5
```

Smoke evidence:

```text
1 fixed VG stress sample completed with relation tokens enabled.
Metadata shows relation tokens such as "plant on side of door" appended after object tokens.
```

Stress-set evidence on the same 18 samples:

```text
relation_tokens vs graph_on mean MAD: 10.99 / 255
relation_tokens vs graph_on median MAD: 10.17 / 255
previous graph_on vs graph_off mean MAD: 0.13 / 255
```

Interpretation:

```text
relation-as-grounding-token is much more effective at changing the generated image than the small GAT residual path.
However, some samples show content/color drift, so the next step should constrain this path rather than simply increasing token count.
```

Diagnostic artifacts:

```text
artifacts/stress_style_gate_on_off_20260704/stress_relation_tokens_vs_graph_on_grid_first12.png
artifacts/stress_style_gate_on_off_20260704/stress_relation_tokens_vs_graph_on_pixel_diff_summary.json
```

## Attention-Box Diagnostic Update

After adding a diagnostic-only attention recorder to GLIGEN's `GatedSelfAttentionDense`
fuser, we ran a clean18 fixed-test analysis on the current compact style-gate
checkpoint.

Protocol:

```text
diagnostic_only_not_formal_eval
split: VG fixed test split
sample_indices: 1,5,14,18,31,36,44,53,56,59,61,62,75,78,124,160,174,185
num_samples: 18
checkpoint: OUTPUT_STANDARD_SG2IM_COMPACT_STYLE_GATE/.../checkpoint_latest.pth
timestep: 500
```

Key attention findings:

```text
64x64 fuser layers: lift ~= 1.00 to 1.08
32x32 fuser layers: lift ~= 1.03 to 1.66
16x16 fuser layers: lift can reach ~= 5.58 to 6.34
middle 8x8 layer: lift ~= 2.29
mean lift across layers: ~= 2.26
```

Here `lift` means:

```text
attention_inside_box_ratio / box_area_ratio
```

Interpretation:

```text
The fuser attention is not completely blind to object boxes.
Some middle/low-resolution layers do concentrate visual queries around the correct boxes.
However, high-resolution 64x64 layers are almost area-random, which is bad for details and small objects.
```

We also inspected the actual GLIGEN fuser gate magnitudes in the loaded checkpoint:

```text
mean_abs(tanh(alpha_attn)) ~= 0.026
max_abs(tanh(alpha_attn)) ~= 0.057
```

This explains the earlier pixel-difference result:

```text
graph_on vs graph_off mean MAD ~= 0.13 / 255
```

The graph/fuser path may compute nontrivial attention, but the residual injected
back into the U-Net is extremely weak. In plain terms:

```text
the model sometimes looks at the right place, but the control signal speaks too quietly
```

Updated next-step implication:

```text
Do not just add more token-space graph losses.
The next useful change should either increase/control the effective fuser injection
or add image-space / attention-space supervision that makes box alignment visible in the generated image.
```

Diagnostic artifact:

```text
artifacts/attention_box_clean18_20260705/attention_box_diagnostic.json
```

## Fuser Alpha Sweep Update

We added an eval-only fuser injection ablation:

```text
FUSER_ALPHA_ATTN_MULTIPLIER
FUSER_ALPHA_DENSE_MULTIPLIER
```

The default is `1.0`, so normal evaluation remains unchanged. The ablation only
changes in-memory fuser alpha parameters during sampling and writes the chosen
multipliers into `meta.txt`.

Clean6 diagnostic setup:

```text
split: VG fixed test split
sample_indices: 1,5,14,18,31,36
sampler: DDIM
steps: 50
guidance: 5.0
dense multiplier: 1.0
attn multipliers: 1.0, 1.5, 2.0, 3.0, 6.0
```

Pixel-difference evidence versus x1:

```text
x1.5 mean MAD ~= 25.6 / 255
x2.0 mean MAD ~= 36.4 / 255
x3.0 mean MAD ~= 51.3 / 255
x6.0 mean MAD ~= 78.9 / 255
```

Visual interpretation:

```text
x1.5 and x2.0 visibly strengthen layout/object organization but already introduce some style drift.
x3.0 makes control much stronger but causes repeated people, hard edges, and unnatural structure.
x6.0 collapses into texture/noise artifacts.
```

Updated conclusion:

```text
The control path can influence the generated image when fuser injection is amplified.
However, naive amplification trades controllability for quality and stability.
The next model change should learn a controlled, layer-aware gate schedule rather than simply multiplying all fuser attention gates.
```

Diagnostic artifacts:

```text
artifacts/fuser_alpha_sweep_clean6_20260705/fuser_alpha_sweep_clean6_grid_full.png
artifacts/fuser_alpha_sweep_clean6_20260705/fuser_alpha_sweep_clean6_pixel_diff_summary_full.json
```

## Layer-Aware Fuser Profile Update

We then added an eval-only layer-aware fuser profile:

```text
FUSER_ALPHA_ATTN_PROFILE="64:1.0,32:1.2,16:1.5,8:1.3"
FUSER_ALPHA_DENSE_PROFILE="all:1.0"
```

and a stronger variant:

```text
FUSER_ALPHA_ATTN_PROFILE="64:1.0,32:1.3,16:1.8,8:1.5"
FUSER_ALPHA_DENSE_PROFILE="all:1.0"
```

The rationale is based on the attention-box diagnostic:

```text
64x64 layers were close to area-random, so they should not be amplified aggressively.
16x16 / 8x8 layers showed more box-aware attention, so they are safer layout-control targets.
```

Clean18 diagnostic evidence:

```text
profile conservative vs x1 mean MAD ~= 26.49 / 255
profile stronger vs x1 mean MAD ~= 31.79 / 255
```

Visual interpretation:

```text
Layer-aware profiles make the control path visibly active.
They are much safer than global x3/x6, which caused texture collapse.
The conservative profile is roughly comparable in magnitude to global x1.5.
The stronger profile is between global x1.5 and global x2.
```

Remaining issue:

```text
The generated images still often follow SD prior more than the exact scene graph.
Some samples still become black-and-white or semantically drift.
Layer-aware scaling improves the control/quality tradeoff, but it does not solve grounding by itself.
```

Updated next-step conclusion:

```text
Do not use a fixed multiplier as the final method.
Use this evidence to motivate a learned or scheduled fuser gate with image-space/attention-space supervision.
The first trainable version should constrain high-resolution layers and emphasize mid/low-resolution spatial layers.
```

Diagnostic artifacts:

```text
artifacts/layer_aware_fuser_clean18_20260705/layer_aware_fuser_clean18_grid_full.png
artifacts/layer_aware_fuser_clean18_20260705/layer_aware_fuser_clean18_pixel_diff_summary_full.json
```
