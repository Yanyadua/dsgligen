# Fresh-Instance Recovery Validation

This record captures the recovery checks run on a fresh AutoDL instance. It
does not claim that lost historical training checkpoints or their metrics were
recreated.

## Restored resources

- Source branch: `recovery/2026-07-28-conditioner`.
- Base checkpoint: official GLIGEN text-box checkpoint at
  `gligen_checkpoints/diffusion_pytorch_model.bin`.
- Checkpoint SHA-256:
  `f5f3d2d5ec6e01c7ad7ca811a39904db675d1c5fccfeca9d34d63e4bf65ccd7b`.
- Clean SG2IM-style VG root: `/root/autodl-tmp/standard_sg2im_fresh_h5`.

## Data protocol validation

`scripts/eval/validate_standard_sg2im_h5.py` passed with:

- train / val / test: `62565 / 5062 / 5096`;
- object and predicate vocab arrays: `179 / 46` including the SG2IM special
  image and in-image symbols;
- train/val, train/test and val/test image-id overlap: zero.

## Runtime validation

- `verify_recovery_bundle.py`: 38 unit tests passed.
- `scripts/smoke_recovery.py`: all recovered model configurations instantiated.
- `scripts/eval/smoke_standard_sg2im_dataset.py`: read representative first,
  middle and final clean-training samples with valid image tensors, normalized
  boxes and remapped relation edges.
- `scripts/eval/audit_historical_vg_run.py`: strict base load passed with 966
  pretrained tensors and 39 scene-graph-only tensors.
- A clean one-step training run completed from the official base checkpoint,
  frozen fuser and frozen compatible PositionNet base; TensorBoard output was
  written.
- A `clean_spatial_v2` one-sample DDIM sampling smoke completed from the fixed
  test H5 and saved real/fake image pairs plus sample metadata.

## Remaining scientific constraints

- The original long-run training checkpoints from released containers are not
  recoverable and are deliberately not substituted by inferred values.
- The one-step training run and one-sample five-step sampler are operational
  checks only. They must not be reported as FID, IS, or paper results.
- Any future metric needs a newly named experiment, fixed train-to-test
  provenance, saved generation metadata, matching real/fake image counts and
  an explicit metric implementation record.
