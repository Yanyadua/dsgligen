import torch


def build_per_sample_noise(
    image_ids,
    sample_shape,
    base_seed,
    device=None,
    dtype=torch.float32,
):
    samples = []
    for image_id in image_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(base_seed) + int(image_id))
        sample = torch.randn(
            tuple(sample_shape),
            generator=generator,
            dtype=dtype,
            device="cpu",
        )
        samples.append(sample)
    noise = torch.stack(samples, dim=0)
    return noise if device is None else noise.to(device)
