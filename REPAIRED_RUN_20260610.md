# Repaired VG Fixed-Split Run, 2026-06-10

## Proven recovery fixes

- Restored the official GLIGEN PositionNet `linears.0/2/4` architecture.
- Enforced strict loading of all 966 compatible official GLIGEN tensors.
- Kept 39 scene-graph-only tensors as the trainable extension.
- Corrected object boxes with the same center-crop and horizontal-flip transform
  applied to training images.
- Enforced the SG2I fixed split: 62,565 train images and 5,096 test images.
- Verified zero train/test image-ID overlap.
- Added strict checkpoint manifests, iteration checks, hashes, and output protocol
  metadata.

## Training evidence before the instance stopped responding

- Clean 1k training completed in `tag00`.
- The 1k checkpoint passed the historical-run audit:
  - 966 official base tensors loaded.
  - 39/39 trainable scene-graph tensors loaded.
  - checkpoint iteration: 1,000.
  - train/test overlap: 0.
- An eight-image qualitative probe completed with:
  - sampler: DDIM.
  - steps: 50.
  - guidance: 5.0.
  - seed: 20260429.
- Training resumed from `tag00/checkpoint_latest.pth` into `tag01`.
- Logs explicitly reported `auto-resumed 39 trainable tensors`.
- Checkpoints through 7,001 were observed before SSH became unavailable.
- At steps 6,000-6,999, TensorBoard thousand-step means were:
  - total loss: 0.21059.
  - diffusion loss: 0.18344.
  - object alignment loss: 0.39403.
  - spatial consistency loss: 0.03850.
  - relation geometry prediction loss: 0.11036.

## Evaluation protocol

- Test split: all 5,096 fixed-split test samples.
- Sampler: DDIM.
- Steps: 50.
- Guidance scale: 5.0.
- Seed: 20260429.
- Evaluation batch size: 4.
- Resolution: 256x256.
- Metrics:
  - `pytorch-fid` 0.3.0, 2048-dimensional Inception features.
  - `torch-fidelity` 0.4.0, FID and Inception Score.

## Restart procedure

Run:

```bash
nohup bash recover_repaired_run_after_instance_restart.sh \
  > recovered_after_restart_pipeline.log 2>&1 < /dev/null &
```

The script:

1. Detects whether the repaired experiment is already training.
2. Uses an existing 10k checkpoint if present.
3. Otherwise resumes the latest lightweight checkpoint to 10k.
4. Audits the final checkpoint and fixed split.
5. Generates an eight-image DDIM50 probe.
6. Generates all 5,096 test images in a checkpoint-hash-specific directory.
7. Verifies exact real/fake counts and protocol metadata.
8. Computes FID/IS using both metric backends.
