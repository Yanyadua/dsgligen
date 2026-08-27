# Layer-Aware Fuser Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eval-only layer-aware fuser alpha profiles and run a small fixed-split diagnostic sweep.

**Architecture:** Extend `scripts/eval/inference_ablation.py` with profile parsing, fuser resolution mapping, and an in-memory profile applicator. Wire the profile environment variables into `scripts/eval/generate_vg_fixedsplit_eval.py` with metadata logging. Validate locally and remotely, then run clean6 sampling.

**Tech Stack:** Python, PyTorch, GLIGEN eval scripts, VG fixed split, unittest/pytest.

---

### Task 1: Add Profile Parser And Applicator

**Files:**
- Modify: `scripts/eval/inference_ablation.py`
- Modify: `tests/test_inference_ablation.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
parse_fuser_alpha_profile("64:1.0,32:1.3,16:1.8,8:1.5")
infer_fuser_resolution("input_blocks.8.1.transformer_blocks.0.fuser")
apply_fuser_alpha_profile(model, attn_profile={64: 1.0, 32: 1.2, 16: 1.5, 8: 1.3})
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest -q tests/test_inference_ablation.py
```

Expected: import or attribute failure for the new profile functions.

- [ ] **Step 3: Implement minimal profile utilities**

Implement:

```python
parse_fuser_alpha_profile(value)
infer_fuser_resolution(name)
resolve_fuser_multiplier(name, profile)
apply_fuser_alpha_profile(model, attn_profile=None, dense_profile=None)
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
pytest -q tests/test_inference_ablation.py
python -m py_compile scripts/eval/inference_ablation.py tests/test_inference_ablation.py
```

Expected: all pass.

### Task 2: Wire Profiles Into Eval Script

**Files:**
- Modify: `scripts/eval/generate_vg_fixedsplit_eval.py`
- Modify: `tests/test_inference_ablation.py`

- [ ] **Step 1: Add environment variables**

Add:

```text
FUSER_ALPHA_ATTN_PROFILE
FUSER_ALPHA_DENSE_PROFILE
```

Default empty string means disabled.

- [ ] **Step 2: Apply after checkpoint load**

Apply profile after the existing global multiplier block. If both global
multiplier and profile are set, the effects compose. This makes debug sweeps
explicit and metadata-tracked.

- [ ] **Step 3: Add metadata fields**

Record exact profile strings in `meta.txt` and sample metadata:

```text
FUSER_ALPHA_ATTN_PROFILE
FUSER_ALPHA_DENSE_PROFILE
```

- [ ] **Step 4: Run syntax and local tests**

Run:

```bash
pytest -q tests/test_inference_ablation.py tests/test_attention_box_metrics.py
python -m py_compile scripts/eval/generate_vg_fixedsplit_eval.py scripts/eval/inference_ablation.py
```

Expected: all pass.

### Task 3: Remote Validation And Clean6 Diagnostic

**Files:**
- Sync: `scripts/eval/inference_ablation.py`
- Sync: `scripts/eval/generate_vg_fixedsplit_eval.py`
- Sync: `tests/test_inference_ablation.py`

- [ ] **Step 1: Run remote unittest**

Run on AutoDL:

```bash
cd /root/autodl-tmp/GLIGEN
/root/miniconda3/bin/python tests/test_inference_ablation.py
```

Expected: `OK`.

- [ ] **Step 2: Run clean6 profiles**

Run DDIM50 on fixed test indices `1,5,14,18,31,36`:

```text
profile_conservative = 64:1.0,32:1.2,16:1.5,8:1.3
profile_stronger = 64:1.0,32:1.3,16:1.8,8:1.5
```

- [ ] **Step 3: Build comparison grid**

Create:

```text
Real | x1 | global x1.5 | profile_conservative | profile_stronger | global x2
```

- [ ] **Step 4: Pull artifacts and document conclusion**

Pull artifacts to:

```text
artifacts/layer_aware_fuser_clean6_20260705/
```

Update:

```text
docs/controlled_generation_code_audit_20260704.md
```

### Task 4: Commit

- [ ] **Step 1: Commit implementation**

Run:

```bash
git add scripts/eval/inference_ablation.py scripts/eval/generate_vg_fixedsplit_eval.py tests/test_inference_ablation.py docs/controlled_generation_code_audit_20260704.md
git commit -m "add layer aware fuser alpha profiles"
```

- [ ] **Step 2: Leave artifacts untracked**

Confirm:

```bash
git status --short
```

Expected: only pre-existing untracked artifacts/scripts remain.
