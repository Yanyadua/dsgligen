from pathlib import Path
import random
import gc
import sys
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from trainer import batch_to_device
from dataset.concat_dataset import ConCatDataset


DEVICE = torch.device("cuda")
OUT = Path("eval_outputs/compare_5k_fixed") / os.environ.get("OUT_NAME", "residual_horse_compare")
OUT.mkdir(parents=True, exist_ok=True)

EXPS = [
    ("baseline", "configs/vg_text_box_baseline.yaml", "OUTPUT_LONG_FIXED/vg_text_box_baseline_5k/tag00/checkpoint_00005000.pth"),
    ("mlp", "configs/vg_scene_graph_mlp.yaml", "OUTPUT_LONG_FIXED/vg_scene_graph_mlp_5k/tag00/checkpoint_00005000.pth"),
    ("gat_old", "configs/vg_scene_graph_gat.yaml", "OUTPUT_LONG_FIXED/vg_scene_graph_gat_5k/tag00/checkpoint_00005000.pth"),
    ("gat_residual", "configs/vg_scene_graph_gat_residual.yaml", "OUTPUT_RESIDUAL/vg_scene_graph_gat_residual_mlpinit_2k/tag00/checkpoint_00002000.pth"),
    ("gat_residual_align", "configs/vg_scene_graph_gat_residual_align.yaml", "OUTPUT_ALIGN/vg_scene_graph_gat_residual_align_mlpinit_2k/tag00/checkpoint_00002000.pth"),
]

BASE_CKPT = "gligen_checkpoints/diffusion_pytorch_model.bin"
SELECTED_INDICES = [int(x) for x in os.environ.get("SAMPLE_INDICES", "15,24,23,34").split(",")]
STEPS = 20
GUIDANCE = 5.0
SEED = 20260421


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_item_batch(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch_to_device(batch, DEVICE)


def load_model(yaml_file, ckpt_file):
    cfg = OmegaConf.load(yaml_file)
    model = instantiate_from_config(cfg.model).to(DEVICE).eval()
    autoencoder = instantiate_from_config(cfg.autoencoder).to(DEVICE).eval()
    text_encoder = instantiate_from_config(cfg.text_encoder).to(DEVICE).eval()
    diffusion = instantiate_from_config(cfg.diffusion).to(DEVICE)

    base = torch.load(BASE_CKPT, map_location="cpu")
    compatible = {
        k: v
        for k, v in base["model"].items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    autoencoder.load_state_dict(base["autoencoder"])
    text_encoder.load_state_dict(base["text_encoder"], strict=False)
    diffusion.load_state_dict(base["diffusion"])

    light = torch.load(ckpt_file, map_location="cpu")
    trainable = light.get("model_trainable")
    assert trainable is not None, f"{ckpt_file} has no model_trainable"
    state = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in trainable.items():
        if k in state and state[k].shape == v.shape:
            state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(state, strict=True)
    print("LOADED", yaml_file, "trainable", loaded, "skipped", len(skipped), flush=True)

    grounding_tokenizer_input = instantiate_from_config(cfg.grounding_tokenizer_input)
    model.grounding_tokenizer_input = grounding_tokenizer_input
    return cfg, model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input


set_seed(SEED)
base_cfg = OmegaConf.load(EXPS[0][1])
base_cfg.DATA_ROOT = "DATA"
base_cfg.train_dataset_names.VGGrounding.prob_use_caption = 1.0
base_cfg.train_dataset_names.VGGrounding.random_flip = False
base_cfg.train_dataset_names.VGGrounding.random_crop = False
dataset = ConCatDataset(base_cfg.train_dataset_names, base_cfg.DATA_ROOT, train=True)
inner = dataset.datasets[0]

items = [dataset[idx] for idx in SELECTED_INDICES]
real_vis = torch.stack(
    [inner.vis_getitem_data(out=item, return_tensor=True, print_caption=False) for item in items],
    dim=0,
)
torchvision.utils.save_image(real_vis, OUT / "conditions_real_boxes.png", nrow=len(items))

with open(OUT / "captions.txt", "w") as f:
    f.write("indices: " + ", ".join(map(str, SELECTED_INDICES)) + "\n\n")
    for i, item in enumerate(items):
        f.write(f"[{i}] dataset_idx={SELECTED_INDICES[i]}\n")
        f.write(f"{item['caption']}\n")
        f.write(f"valid_boxes={int(item['masks'].sum().item())}\n\n")
print("USING_INDICES", SELECTED_INDICES, flush=True)

noise_shape = (1, 4, 64, 64)
set_seed(SEED + 99)
noises = [torch.randn(noise_shape, device=DEVICE) for _ in items]
sample_paths = []

for label, yaml_file, ckpt_file in EXPS:
    print("SAMPLING", label, flush=True)
    cfg, model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input = load_model(yaml_file, ckpt_file)
    sampler = PLMSSampler(diffusion, model)
    decoded_list = []
    with torch.no_grad():
        for i, item in enumerate(items):
            batch = one_item_batch(item)
            context = text_encoder.encode(batch["caption"])
            uc = text_encoder.encode([""])
            grounding_input = grounding_tokenizer_input.prepare(batch)
            input_dict = dict(
                x=noises[i].clone(),
                timesteps=None,
                context=context,
                grounding_input=grounding_input,
                inpainting_extra_input=None,
                grounding_extra_input=None,
            )
            print("  item", i, "idx", SELECTED_INDICES[i], "caption", batch["caption"][0][:100], flush=True)
            samples = sampler.sample(S=STEPS, shape=noise_shape, input=input_dict, uc=uc, guidance_scale=GUIDANCE)
            decoded = autoencoder.decode(samples).cpu()
            decoded = torch.clamp(decoded, min=-1, max=1)
            decoded_list.append(decoded[0])
            del batch, context, uc, grounding_input, input_dict, samples, decoded
            torch.cuda.empty_cache()
    decoded_all = torch.stack(decoded_list, dim=0)
    out_path = OUT / f"{label}_samples.png"
    torchvision.utils.save_image(
        decoded_all,
        out_path,
        nrow=len(items),
        normalize=True,
        scale_each=True,
        value_range=(-1, 1),
    )
    sample_paths.append((label, out_path))
    del cfg, model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input, sampler, decoded_all, decoded_list
    torch.cuda.empty_cache()
    gc.collect()

rows = [("condition", OUT / "conditions_real_boxes.png")] + sample_paths
imgs = []
for label, path in rows:
    img = Image.open(path).convert("RGB")
    pad = 40
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), label, fill=(0, 0, 0))
    imgs.append(canvas)

width = max(im.width for im in imgs)
height = sum(im.height for im in imgs)
sheet = Image.new("RGB", (width, height), "white")
y = 0
for im in imgs:
    sheet.paste(im, (0, y))
    y += im.height
sheet.save(OUT / "residual_comparison_sheet.png")
print("SAVED", OUT, flush=True)
