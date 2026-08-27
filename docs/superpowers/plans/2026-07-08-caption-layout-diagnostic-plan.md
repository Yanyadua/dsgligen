# Caption Layout Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-way clean18 diagnostic that separates GLIGEN base quality, caption wording, trained checkpoint damage, and graph-branch effect.

**Architecture:** Add a deterministic caption policy switch to the existing fixed-split evaluation script. Keep the dataset, checkpoint, sampler, split, object selection, and random noise policy fixed across runs so only one intended variable changes per comparison.

**Tech Stack:** Python, PyTorch, GLIGEN evaluation script, Visual Genome SG2I fixed split, DDIM50.

---

### Task 1: Add Caption Policy Switch

**Files:**
- Modify: `dataset/scene_graph_caption.py`
- Modify: `scripts/eval/generate_vg_fixedsplit_eval.py`
- Test: `tests/test_scene_graph_caption.py`

- [ ] Add `build_natural_scene_graph_caption(...)` beside the existing graph caption builder.
- [ ] Add `CAPTION_POLICY=os.environ.get("CAPTION_POLICY", "graph")` to the eval script.
- [ ] Route `_caption_from_graph(...)` to either `graph` or `natural`.
- [ ] Record the active policy in `meta.txt`.
- [ ] Test that `graph` preserves the old mechanical relation sentence and `natural` produces one fluent sentence.

### Task 2: Add Four-Way Diagnostic Launcher

**Files:**
- Create: `run_clean18_caption_graph_diagnostic.sh`

- [ ] Run A: official GLIGEN base, natural caption, graph branch absent.
- [ ] Run B: official GLIGEN base, graph caption, graph branch absent.
- [ ] Run C: compact style-gate checkpoint, graph caption, graph gate forced off.
- [ ] Run D: compact style-gate checkpoint, graph caption, graph gate default on.
- [ ] Use SG2I fixed test split, clean18 indices, DDIM50, guidance 3.0, batch size 4.

### Task 3: Verify And Run

**Files:**
- No code changes.

- [ ] Run local unit tests with `python tests/test_scene_graph_caption.py`.
- [ ] Sync changed files to AutoDL.
- [ ] Run remote unit tests.
- [ ] Run `run_clean18_caption_graph_diagnostic.sh` remotely.
- [ ] Pull output images and build a real/A/B/C/D grid.
