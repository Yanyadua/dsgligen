import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from convert_ckpt import add_additional_channels
import math
import re
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
try:
    from apex import amp
except:
    pass  
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #



class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, masked_real, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def normalize_collated_text_grid(value, batch_size):
    """Default PyTorch collation transposes fixed-length string lists."""
    if value is None:
        return None
    if len(value) == 0:
        return []
    if len(value) == batch_size and all(isinstance(row, (list, tuple)) for row in value):
        return [list(row) for row in value]
    return [list(items) for items in zip(*value)]


GEOMETRIC_RELATION_PATTERNS = {
    "left": re.compile(r"\b(left|left of|to the left of)\b"),
    "right": re.compile(r"\b(right|right of|to the right of)\b"),
    "above": re.compile(r"\b(above|over|on top of)\b"),
    "below": re.compile(r"\b(below|under|beneath)\b"),
    "inside": re.compile(r"\b(in|inside|within)\b"),
    "on": re.compile(r"\b(on|upon|sitting on|standing on|lying on)\b"),
}


def relation_text_geometry_mask(relation_texts, batch_size, device, dtype):
    """Select relation labels whose meaning is primarily geometric."""
    normalized = normalize_collated_text_grid(relation_texts, batch_size)
    if not normalized:
        return None
    mask = torch.zeros((batch_size, len(normalized[0])), device=device, dtype=dtype)
    for batch_idx, row in enumerate(normalized):
        for rel_idx, text in enumerate(row):
            text = str(text).lower()
            if any(pattern.search(text) for pattern in GEOMETRIC_RELATION_PATTERNS.values()):
                mask[batch_idx, rel_idx] = 1
    return mask


def corrupt_relation_geo_features(relation_geo_features):
    wrong_geo = relation_geo_features.clone()
    wrong_geo[..., 0] = -wrong_geo[..., 0]
    wrong_geo[..., 1] = -wrong_geo[..., 1]
    wrong_geo[..., 8], wrong_geo[..., 9] = relation_geo_features[..., 9], relation_geo_features[..., 8]
    wrong_geo[..., 10], wrong_geo[..., 11] = relation_geo_features[..., 11], relation_geo_features[..., 10]
    return wrong_geo


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 






class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        self.graph_stats_interval = int(getattr(config, "graph_stats_interval", 50))
        self.graph_stats_enabled = bool(getattr(config, "log_graph_stats", True))
        self.graph_param_snapshot = None
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        
        init_from_gligen_ckpt = getattr(config, "init_from_gligen_ckpt", None)
        if init_from_gligen_ckpt is not None:
            if get_rank() == 0:
                print("initializing from GLIGEN checkpoint " + init_from_gligen_ckpt)
            state_dict = torch.load(init_from_gligen_ckpt, map_location="cpu")
            for key in ["model", "autoencoder", "text_encoder", "diffusion"]:
                assert key in state_dict, f"{init_from_gligen_ckpt} is missing {key}"
        else:
            state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        
        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels>0 else False

        # load original SD/GLIGEN ckpt (with input conv may be modified)
        if init_from_gligen_ckpt is not None:
            compatible_model_state = {
                k: v for k, v in state_dict["model"].items()
                if (k in self.model.state_dict() and self.model.state_dict()[k].shape == v.shape)
            }
            missing_keys, unexpected_keys = self.model.load_state_dict(compatible_model_state, strict=False)
            if get_rank() == 0:
                skipped = len(state_dict["model"]) - len(compatible_model_state)
                print(f"loaded {len(compatible_model_state)} compatible model tensors, skipped {skipped}")
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
            assert unexpected_keys == []
        original_params_names = list( self.model.state_dict().keys()  ) # used for sanity check later
        
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"], strict=False  )
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)

        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            self.model.load_state_dict(first_stage_ckpt["model"])

        grounding_ckpt = getattr(self.config, "grounding_ckpt", None)
        if grounding_ckpt is not None:
            grounding_state = torch.load(grounding_ckpt, map_location="cpu")
            grounding_state = grounding_state.get("model_trainable", grounding_state.get("model", {}))
            current_state = self.model.state_dict()
            compatible_grounding = {
                k: v for k, v in grounding_state.items()
                if k in current_state and current_state[k].shape == v.shape
            }
            current_state.update(compatible_grounding)
            self.model.load_state_dict(current_state, strict=True)
            if get_rank() == 0:
                skipped = len(grounding_state) - len(compatible_grounding)
                print(f"loaded {len(compatible_grounding)} compatible grounding tensors from {grounding_ckpt}, skipped {skipped}")


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        freeze_position_base = getattr(config, "freeze_position_base", False)
        for name, p in self.model.named_parameters():
            if ("transformer_blocks" in name) and ("fuser" in name):
                # New added Attention layers. Freeze for encoder-only ablations.
                if not getattr(config, "freeze_fuser", False):
                    params.append(p)
                    trainable_names.append(name)
            elif  "position_net" in name:
                # For graph-adapter ablations, keep the object/box MLP path fixed
                # and train only the scene-graph residual branch.
                if freeze_position_base:
                    is_graph_adapter_param = (
                        "position_net.gat_layers" in name
                        or "position_net.graph_gate" in name
                        or "position_net.graph_adapter" in name
                        or "position_net.relation_geo_predictor" in name
                    )
                    if is_graph_adapter_param:
                        params.append(p)
                        trainable_names.append(name)
                    else:
                        p.requires_grad = False
                else:
                    # Grounding token processing network
                    params.append(p)
                    trainable_names.append(name)
            elif  "downsample_net" in name:
                # Grounding downsample network (used in input) 
                params.append(p) 
                trainable_names.append(name)
            elif (self.input_conv_train) and ("input_blocks.0.0.weight" in name):
                # First conv layer was modified, thus need to train 
                params.append(p) 
                trainable_names.append(name)
            else:
                # Following make sure we do not miss any new params
                # all new added trainable params have to be haddled above
                # otherwise it will trigger the following error  
                assert name in original_params_names, name 
            all_params_name.append(name) 


        self.trainable_names = trainable_names
        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
        
        


        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()




        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 




        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)     
        



        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()


        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )

    def get_graph_named_parameters(self):
        model_wo_wrapper = self.model.module if self.config.distributed else self.model
        position_net = getattr(model_wo_wrapper, "position_net", None)
        if position_net is None:
            return []

        graph_prefixes = ("gat_layers.", "graph_gate", "graph_adapter.", "relation_geo_predictor.")
        named_params = []
        for name, param in position_net.named_parameters():
            if any(name.startswith(prefix) for prefix in graph_prefixes):
                named_params.append((name, param))
        return named_params

    def snapshot_graph_parameters(self):
        if not self.graph_stats_enabled:
            return
        self.graph_param_snapshot = {
            name: param.detach().float().cpu().clone()
            for name, param in self.get_graph_named_parameters()
        }

    def log_graph_stats(self):
        if (not self.graph_stats_enabled) or self.writer is None:
            return

        named_params = self.get_graph_named_parameters()
        if len(named_params) == 0:
            return

        param_norm_sq = 0.0
        grad_norm_sq = 0.0
        update_norm_sq = 0.0
        grad_abs_mean_values = []
        update_abs_mean_values = []

        for name, param in named_params:
            param_data = param.detach().float()
            param_norm_sq += float(param_data.pow(2).sum().item())

            if param.grad is not None:
                grad_data = param.grad.detach().float()
                grad_norm_sq += float(grad_data.pow(2).sum().item())
                grad_abs_mean_values.append(float(grad_data.abs().mean().item()))

            if self.graph_param_snapshot is not None and name in self.graph_param_snapshot:
                prev = self.graph_param_snapshot[name].to(param_data.device)
                delta = param_data - prev
                update_norm_sq += float(delta.pow(2).sum().item())
                update_abs_mean_values.append(float(delta.abs().mean().item()))

        self.writer.add_scalar("graph_stats/param_norm", math.sqrt(max(param_norm_sq, 0.0)), self.iter_idx + 1)
        self.writer.add_scalar("graph_stats/grad_norm", math.sqrt(max(grad_norm_sq, 0.0)), self.iter_idx + 1)
        self.writer.add_scalar("graph_stats/update_norm", math.sqrt(max(update_norm_sq, 0.0)), self.iter_idx + 1)

        if grad_abs_mean_values:
            self.writer.add_scalar(
                "graph_stats/grad_abs_mean",
                sum(grad_abs_mean_values) / len(grad_abs_mean_values),
                self.iter_idx + 1,
            )
        if update_abs_mean_values:
            self.writer.add_scalar(
                "graph_stats/update_abs_mean",
                sum(update_abs_mean_values) / len(update_abs_mean_values),
                self.iter_idx + 1,
            )

        model_wo_wrapper = self.model.module if self.config.distributed else self.model
        position_net = getattr(model_wo_wrapper, "position_net", None)
        graph_gate = getattr(position_net, "graph_gate", None) if position_net is not None else None
        if graph_gate is not None:
            self.writer.add_scalar(
                "graph_stats/graph_gate_sigmoid",
                torch.sigmoid(graph_gate.detach()).item(),
                self.iter_idx + 1,
            )

        for name, param in named_params:
            safe_name = name.replace(".", "/")
            self.writer.add_scalar(
                f"graph_stats/per_param/{safe_name}_param_norm",
                param.detach().float().norm().item(),
                self.iter_idx + 1,
            )
            if param.grad is not None:
                self.writer.add_scalar(
                    f"graph_stats/per_param/{safe_name}_grad_norm",
                    param.grad.detach().float().norm().item(),
                    self.iter_idx + 1,
                )


    @torch.no_grad()
    def encode_grounding_text_features(self, texts):
        _, pooler_output = self.text_encoder.encode(texts, return_pooler_output=True)
        return pooler_output


    @torch.no_grad()
    def ensure_scene_graph_text_embeddings(self, batch):
        batch_size = batch["image"].shape[0]

        object_texts = normalize_collated_text_grid(batch.get("object_texts"), batch_size)
        if object_texts:
            flat_object_texts = [text if text else "" for row in object_texts for text in row]
            object_embeddings = self.encode_grounding_text_features(flat_object_texts)
            batch["text_embeddings"] = object_embeddings.view(batch_size, len(object_texts[0]), -1)

        relation_texts = normalize_collated_text_grid(batch.get("relation_texts"), batch_size)
        if relation_texts:
            flat_relation_texts = [text if text else "" for row in relation_texts for text in row]
            relation_embeddings = self.encode_grounding_text_features(flat_relation_texts)
            batch["relation_embeddings"] = relation_embeddings.view(batch_size, len(relation_texts[0]), -1)




    @torch.no_grad()
    def get_input(self, batch):

        z = self.autoencoder.encode( batch["image"] )

        context = self.text_encoder.encode( batch["caption"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
            masked_z = z*inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, t, context, inpainting_extra_input, grounding_extra_input 


    def run_one_step(self, batch):
        self.ensure_scene_graph_text_embeddings(batch)
        x_start, t, context, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)
        model_output = self.model(input)
        
        diffusion_loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight
        loss = diffusion_loss
        self.loss_dict = {"loss": loss.item(), "diffusion_loss": diffusion_loss.item()}

        object_align_loss_weight = getattr(self.config, "object_align_loss_weight", 0.0)
        if object_align_loss_weight > 0:
            object_tokens = self.model.position_net(**grounding_input)
            positive_embeddings = grounding_input["positive_embeddings"].detach()
            masks = grounding_input["masks"].to(dtype=object_tokens.dtype)
            object_tokens = torch.nn.functional.normalize(object_tokens, dim=-1)
            positive_embeddings = torch.nn.functional.normalize(positive_embeddings, dim=-1)
            per_object_align = 1 - (object_tokens * positive_embeddings).sum(dim=-1)
            object_align_loss = (per_object_align * masks).sum() / masks.sum().clamp(min=1)
            loss = loss + object_align_loss_weight * object_align_loss
            self.loss_dict = {
                "loss": loss.item(),
                "diffusion_loss": diffusion_loss.item(),
                "object_align_loss": object_align_loss.item(),
            }

        spatial_consistency_loss_weight = getattr(self.config, "spatial_consistency_loss_weight", 0.0)
        if spatial_consistency_loss_weight > 0:
            spatial_margin = getattr(self.config, "spatial_consistency_margin", 0.05)
            positive_embeddings = grounding_input["positive_embeddings"].detach()
            masks = grounding_input["masks"].to(dtype=positive_embeddings.dtype)

            correct_tokens = self.model.position_net(**grounding_input)
            shuffled_input = dict(grounding_input)
            shuffled_input["boxes"] = torch.roll(grounding_input["boxes"], shifts=1, dims=1)
            wrong_tokens = self.model.position_net(**shuffled_input)

            correct_tokens = torch.nn.functional.normalize(correct_tokens, dim=-1)
            wrong_tokens = torch.nn.functional.normalize(wrong_tokens, dim=-1)
            positive_embeddings = torch.nn.functional.normalize(positive_embeddings, dim=-1)
            positive_score = (correct_tokens * positive_embeddings).sum(dim=-1)
            negative_score = (wrong_tokens * positive_embeddings).sum(dim=-1)
            per_object_spatial = torch.relu(spatial_margin + negative_score - positive_score)
            spatial_consistency_loss = (per_object_spatial * masks).sum() / masks.sum().clamp(min=1)
            loss = loss + spatial_consistency_loss_weight * spatial_consistency_loss
            self.loss_dict.update({
                "loss": loss.item(),
                "spatial_consistency_loss": spatial_consistency_loss.item(),
            })

        object_box_contrastive_loss_weight = getattr(self.config, "object_box_contrastive_loss_weight", 0.0)
        if object_box_contrastive_loss_weight > 0:
            temperature = getattr(self.config, "contrastive_temperature", 0.07)
            num_negatives = int(getattr(self.config, "object_box_contrastive_negatives", 3))
            positive_embeddings = torch.nn.functional.normalize(grounding_input["positive_embeddings"].detach(), dim=-1)
            masks = grounding_input["masks"].to(dtype=positive_embeddings.dtype)
            correct_tokens = torch.nn.functional.normalize(self.model.position_net(**grounding_input), dim=-1)
            positive_score = (correct_tokens * positive_embeddings).sum(dim=-1, keepdim=True)

            negative_scores = []
            for shift in range(1, num_negatives + 1):
                shuffled_input = dict(grounding_input)
                shuffled_input["boxes"] = torch.roll(grounding_input["boxes"], shifts=shift, dims=1)
                wrong_tokens = torch.nn.functional.normalize(self.model.position_net(**shuffled_input), dim=-1)
                negative_scores.append((wrong_tokens * positive_embeddings).sum(dim=-1, keepdim=True))

            logits = torch.cat([positive_score] + negative_scores, dim=-1) / temperature
            labels = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
            per_object_contrast = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="none",
            ).view_as(labels)
            object_box_contrastive_loss = (per_object_contrast * masks).sum() / masks.sum().clamp(min=1)
            loss = loss + object_box_contrastive_loss_weight * object_box_contrastive_loss
            self.loss_dict.update({
                "loss": loss.item(),
                "object_box_contrastive_loss": object_box_contrastive_loss.item(),
            })

        relation_contrastive_loss_weight = getattr(self.config, "relation_contrastive_loss_weight", 0.0)
        if relation_contrastive_loss_weight > 0 and "relation_edges" in grounding_input and "relation_embeddings" in grounding_input:
            temperature = getattr(self.config, "contrastive_temperature", 0.07)
            relation_edges = grounding_input["relation_edges"].long()
            relation_embeddings = grounding_input["relation_embeddings"].detach()
            relation_masks = grounding_input.get("relation_masks")
            if relation_masks is not None and relation_edges.shape[1] > 1:
                object_tokens = torch.nn.functional.normalize(self.model.position_net(**grounding_input), dim=-1)
                relation_embeddings = torch.nn.functional.normalize(relation_embeddings, dim=-1)
                node_count = object_tokens.shape[1]
                edge_index = relation_edges.clamp(min=0, max=max(node_count - 1, 0))
                src = edge_index[..., 0]
                dst = edge_index[..., 1]
                batch_idx = torch.arange(object_tokens.shape[0], device=object_tokens.device)[:, None].expand_as(src)
                pair_tokens = torch.nn.functional.normalize(
                    object_tokens[batch_idx, src] + object_tokens[batch_idx, dst],
                    dim=-1,
                )
                logits = torch.matmul(pair_tokens, relation_embeddings.transpose(1, 2)) / temperature
                labels = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)
                per_relation_contrast = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).view(logits.shape[:2])
                relation_masks = relation_masks.to(dtype=per_relation_contrast.dtype)
                relation_contrastive_loss = (per_relation_contrast * relation_masks).sum() / relation_masks.sum().clamp(min=1)
                loss = loss + relation_contrastive_loss_weight * relation_contrastive_loss
                self.loss_dict.update({
                    "loss": loss.item(),
                    "relation_contrastive_loss": relation_contrastive_loss.item(),
                })

        relation_geo_consistency_loss_weight = getattr(self.config, "relation_geo_consistency_loss_weight", 0.0)
        has_relation_geo_inputs = (
            "relation_edges" in grounding_input
            and "relation_embeddings" in grounding_input
            and "relation_geo_features" in grounding_input
        )
        if relation_geo_consistency_loss_weight > 0 and has_relation_geo_inputs:
            relation_edges = grounding_input["relation_edges"].long()
            relation_masks = grounding_input.get("relation_masks")
            relation_embeddings = grounding_input["relation_embeddings"].detach()
            if relation_masks is not None and relation_edges.shape[1] > 0:
                margin = getattr(self.config, "relation_geo_consistency_margin", 0.05)
                filter_predicates = bool(getattr(self.config, "relation_geo_consistency_filter_predicates", True))

                correct_tokens = torch.nn.functional.normalize(self.model.position_net(**grounding_input), dim=-1)
                corrupted_input = dict(grounding_input)
                corrupted_input["relation_geo_features"] = corrupt_relation_geo_features(
                    grounding_input["relation_geo_features"]
                )
                wrong_tokens = torch.nn.functional.normalize(self.model.position_net(**corrupted_input), dim=-1)

                relation_embeddings = torch.nn.functional.normalize(relation_embeddings, dim=-1)
                node_count = correct_tokens.shape[1]
                edge_index = relation_edges.clamp(min=0, max=max(node_count - 1, 0))
                src = edge_index[..., 0]
                dst = edge_index[..., 1]
                batch_idx = torch.arange(correct_tokens.shape[0], device=correct_tokens.device)[:, None].expand_as(src)

                correct_pairs = torch.nn.functional.normalize(
                    correct_tokens[batch_idx, src] + correct_tokens[batch_idx, dst],
                    dim=-1,
                )
                wrong_pairs = torch.nn.functional.normalize(
                    wrong_tokens[batch_idx, src] + wrong_tokens[batch_idx, dst],
                    dim=-1,
                )
                positive_score = (correct_pairs * relation_embeddings).sum(dim=-1)
                negative_score = (wrong_pairs * relation_embeddings).sum(dim=-1)
                per_relation_geo = torch.relu(margin + negative_score - positive_score)

                relation_masks = relation_masks.to(dtype=per_relation_geo.dtype)
                if filter_predicates:
                    geometry_mask = relation_text_geometry_mask(
                        batch.get("relation_texts"),
                        batch_size=correct_tokens.shape[0],
                        device=per_relation_geo.device,
                        dtype=per_relation_geo.dtype,
                    )
                    if geometry_mask is not None and (geometry_mask * relation_masks).sum() > 0:
                        relation_masks = relation_masks * geometry_mask

                relation_geo_consistency_loss = (
                    per_relation_geo * relation_masks
                ).sum() / relation_masks.sum().clamp(min=1)
                loss = loss + relation_geo_consistency_loss_weight * relation_geo_consistency_loss
                self.loss_dict.update({
                    "loss": loss.item(),
                    "relation_geo_consistency_loss": relation_geo_consistency_loss.item(),
                })

        relation_geo_prediction_loss_weight = getattr(self.config, "relation_geo_prediction_loss_weight", 0.0)
        position_net = self.model.position_net
        has_relation_geo_predictor = (
            hasattr(position_net, "predict_relation_geo")
            and getattr(position_net, "relation_geo_predictor", None) is not None
        )
        if relation_geo_prediction_loss_weight > 0 and has_relation_geo_inputs and has_relation_geo_predictor:
            relation_edges = grounding_input["relation_edges"].long()
            relation_masks = grounding_input.get("relation_masks")
            relation_embeddings = grounding_input["relation_embeddings"].detach()
            relation_geo_features = grounding_input["relation_geo_features"].detach()
            if relation_masks is not None and relation_edges.shape[1] > 0:
                object_tokens = self.model.position_net(**grounding_input)
                pred_geo = self.model.position_net.predict_relation_geo(
                    object_tokens,
                    relation_edges,
                    relation_embeddings=relation_embeddings,
                )
                target_geo = relation_geo_features.to(dtype=pred_geo.dtype)
                beta = getattr(self.config, "relation_geo_prediction_beta", 0.1)
                per_relation_pred = torch.nn.functional.smooth_l1_loss(
                    pred_geo,
                    target_geo,
                    reduction="none",
                    beta=beta,
                ).mean(dim=-1)

                relation_masks = relation_masks.to(dtype=per_relation_pred.dtype)
                if bool(getattr(self.config, "relation_geo_prediction_filter_predicates", True)):
                    geometry_mask = relation_text_geometry_mask(
                        batch.get("relation_texts"),
                        batch_size=object_tokens.shape[0],
                        device=per_relation_pred.device,
                        dtype=per_relation_pred.dtype,
                    )
                    if geometry_mask is not None and (geometry_mask * relation_masks).sum() > 0:
                        relation_masks = relation_masks * geometry_mask

                relation_geo_prediction_loss = (
                    per_relation_pred * relation_masks
                ).sum() / relation_masks.sum().clamp(min=1)
                loss = loss + relation_geo_prediction_loss_weight * relation_geo_prediction_loss
                self.loss_dict.update({
                    "loss": loss.item(),
                    "relation_geo_prediction_loss": relation_geo_prediction_loss.item(),
                })

        return loss 
        


    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            should_log_graph_stats = self.graph_stats_enabled and (iter_idx % self.graph_stats_interval == 0)
            if should_log_graph_stats:
                self.snapshot_graph_parameters()
            loss.backward()
            self.opt.step() 
            self.scheduler.step()
            if should_log_graph_stats and get_rank() == 0:
                self.log_graph_stats()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (not getattr(self.config, "disable_saving_in_training", False)) and ((iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1)):
                    self.save_ckpt_and_result()
            synchronize()

        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)
            self.ensure_scene_graph_text_embeddings(batch)

            
            if "boxes" in batch:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)
                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            else:
                # keypoint case 
                real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
                
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            # extra input for inpainting 
            inpainting_extra_input = None
            if self.config.inpaint_mode:
                z = self.autoencoder.encode( batch["image"] )
                inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
                masked_z = z*inpainting_mask
                inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
            
            grounding_extra_input = None
            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          inpainting_extra_input=inpainting_extra_input,
                          grounding_extra_input=grounding_extra_input,
                          grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            masked_real_image =  batch["image"]*torch.nn.functional.interpolate(inpainting_mask, size=(512, 512)) if self.config.inpaint_mode else None
            self.image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, batch["caption"], iter_name)

        if getattr(self.config, "save_trainable_only", False):
            full_model_state = model_wo_wrapper.state_dict()
            trainable_name_set = set(self.trainable_names)
            model_trainable = {
                k: v.detach().cpu()
                for k, v in full_model_state.items()
                if k in trainable_name_set
            }
            ckpt = dict(model_trainable=model_trainable,
                        opt=self.opt.state_dict(),
                        scheduler=self.scheduler.state_dict(),
                        iters=self.iter_idx+1,
                        config_dict=self.config_dict,
                        trainable_names=self.trainable_names)
        else:
            ckpt = dict(model = model_wo_wrapper.state_dict(),
                        text_encoder = self.text_encoder.state_dict(),
                        autoencoder = self.autoencoder.state_dict(),
                        diffusion = self.diffusion.state_dict(),
                        opt = self.opt.state_dict(),
                        scheduler= self.scheduler.state_dict(),
                        iters = self.iter_idx+1,
                        config_dict=self.config_dict,
            )
            if self.config.enable_ema:
                ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )
