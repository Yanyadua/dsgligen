# Trainable Layer Gate + Attention-Box Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest trainable version of layer-aware fuser gating with an attention-box loss for short-run VG controllability experiments.

**Architecture:** Keep the existing GLIGEN fuser structure and add two safe extensions: differentiable attention recording in `SelfAttention/GatedSelfAttentionDense`, and a trainer-side attention-box loss that reads recorded fuser attention after the diffusion forward pass. Use config to freeze most modules and train only fuser gates/linear plus existing scene-graph branch.

**Tech Stack:** Python, PyTorch, GLIGEN attention blocks, VG fixed split, pytest/unittest.

---

### Task 1: Make Fuser Attention Recording Differentiable When Requested

**Files:**
- Modify: `ldm/modules/attention.py`
- Modify: `tests/test_attention_recorder.py`

- [ ] **Step 1: Add failing tests**

Add a test that calls:

```python
layer.set_attention_recording(True, detach=False)
out = layer(x)
recorded = layer.get_last_attention()
assert recorded.requires_grad
```

Expected before implementation: `TypeError` because `detach` argument is not supported.

- [ ] **Step 2: Implement minimal recorder flag**

Add `record_attention_detached=True` and update `set_attention_recording(enabled=True, detach=True)`.

- [ ] **Step 3: Verify**

Run:

```bash
pytest -q tests/test_attention_recorder.py
python -m py_compile ldm/modules/attention.py tests/test_attention_recorder.py
```

### Task 2: Add Attention-Box Loss Helper

**Files:**
- Create: `ldm/modules/attention_box_loss.py`
- Create: `tests/test_attention_box_loss.py`

- [ ] **Step 1: Add failing tests**

Test that a concentrated attention map has lower loss than uniform attention for the same box:

```python
loss_good < loss_uniform
```

- [ ] **Step 2: Implement helper**

Implement:

```python
compute_attention_box_loss_from_attention(visual_to_grounding_attention, boxes, masks)
collect_fuser_attention_box_loss(model, boxes, masks, layer_weights=None)
set_fuser_attention_recording(model, enabled, detach)
```

The loss is:

```text
mean(valid_tokens, relu(target_inside_ratio - inside_ratio))
```

Default `target_inside_ratio=0.5`.

### Task 3: Wire Loss Into Trainer

**Files:**
- Modify: `trainer.py`

- [ ] **Step 1: Add config gate**

Read:

```text
attention_box_loss_weight
attention_box_loss_target
attention_box_loss_layer_weights
```

- [ ] **Step 2: Record attention around model forward**

If weight > 0:

```python
set_fuser_attention_recording(self.model, True, detach=False)
model_output = self.model(input)
attention_box_loss = collect_fuser_attention_box_loss(...)
set_fuser_attention_recording(self.model, False, detach=True)
```

- [ ] **Step 3: Add to loss dict**

Record:

```text
attention_box_loss
attention_box_loss_weighted
```

### Task 4: Add Short-Run Config

**Files:**
- Create: `configs/vg_standard_sg2im_scene_graph_layergate_attnbox_short.yaml`

Base it on `configs/vg_standard_sg2im_scene_graph_compact_style_gate.yaml`, with:

```text
model.params.use_checkpoint: False
fuser_train_mode: gates_and_linear
attention_box_loss_weight: 0.02
attention_box_loss_target: 0.5
save_trainable_only: true
total_iters: 500
```

This config is for smoke/short-run only, not final reporting.

### Task 5: Remote Smoke

**Files:**
- Sync modified code and config to `/root/autodl-tmp/GLIGEN`.

Run a 1-step train smoke with fixed train split. Confirm:

```text
attention_box_loss is finite
diffusion_loss is finite
checkpoint save path is unique
```

Then decide whether to run 300 steps.
