import itertools
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from mixofshow.models.edlora import (LoRALinearLayer, revise_edlora_unet_attention_controller_forward,
                                     revise_edlora_unet_attention_forward)
from mixofshow.pipelines.pipeline_edlora import bind_concept_prompt
from mixofshow.utils.ptp_util import AttentionStore

class EDLoRATrainer(nn.Module):
    def __init__(
        self,
        pretrained_path,
        new_concept_token,
        initializer_token,
        enable_edlora,  # true for ED-LoRA, false for LoRA
        finetune_cfg=None,
        noise_offset=None,
        attn_reg_weight=None,
        reg_full_identity=True,  # True for thanos, False for real person (don't need to encode clothes)
        use_mask_loss=True,
        enable_xformers=False,
        gradient_checkpoint=False,
        replace_mapping=None,  # Add replace_mapping parameter
    ):
        super().__init__()

        # 1. Load the model.
        self.vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder='vae')
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder='unet')

        if gradient_checkpoint:
            self.unet.enable_gradient_checkpointing()

        if enable_xformers:
            assert is_xformers_available(), 'need to install xformer first'

        # 2. Define train scheduler
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder='scheduler')

        # Store replace_mapping if provided
        self.replace_mapping = replace_mapping

        # 3. define training cfg
        self.enable_edlora = enable_edlora
        self.new_concept_cfg = self.init_new_concept(new_concept_token, initializer_token, enable_edlora=enable_edlora)

        self.attn_reg_weight = attn_reg_weight
        self.reg_full_identity = reg_full_identity
        if self.attn_reg_weight is not None:
            self.controller = AttentionStore(training=True)
            revise_edlora_unet_attention_controller_forward(self.unet, self.controller)  # support both lora and edlora forward
        else:
            revise_edlora_unet_attention_forward(self.unet)  # support both lora and edlora forward

        if finetune_cfg:
            self.set_finetune_cfg(finetune_cfg)

        self.noise_offset = noise_offset
        self.use_mask_loss = use_mask_loss

    def set_finetune_cfg(self, finetune_cfg):
        logger = get_logger('mixofshow', log_level='INFO')
        params_to_freeze = [self.vae.parameters(), self.text_encoder.parameters(), self.unet.parameters()]

        # step 1: close all parameters, required_grad to False
        for params in itertools.chain(*params_to_freeze):
            params.requires_grad = False

        # step 2: begin to add trainable paramters
        params_group_list = []

        # 1. text embedding
        if finetune_cfg['text_embedding']['enable_tuning']:
            text_embedding_cfg = finetune_cfg['text_embedding']

            params_list = []
            for params in self.text_encoder.get_input_embeddings().parameters():
                params.requires_grad = True
                params_list.append(params)

            params_group = {'params': params_list, 'lr': text_embedding_cfg['lr']}
            if 'weight_decay' in text_embedding_cfg:
                params_group.update({'weight_decay': text_embedding_cfg['weight_decay']})
            params_group_list.append(params_group)
            logger.info(f"optimizing embedding using lr: {text_embedding_cfg['lr']}")

        # 2. text encoder
        if finetune_cfg['text_encoder']['enable_tuning'] and finetune_cfg['text_encoder'].get('lora_cfg'):
            text_encoder_cfg = finetune_cfg['text_encoder']

            where = text_encoder_cfg['lora_cfg'].pop('where')
            assert where in ['CLIPEncoderLayer', 'CLIPAttention']

            self.text_encoder_lora = nn.ModuleList()
            params_list = []

            for name, module in self.text_encoder.named_modules():
                if module.__class__.__name__ == where:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == 'Linear':
                            lora_module = LoRALinearLayer(name + '.' + child_name, child_module, **text_encoder_cfg['lora_cfg'])
                            self.text_encoder_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': text_encoder_cfg['lr']})
            logger.info(f"optimizing text_encoder ({len(self.text_encoder_lora)} LoRAs), using lr: {text_encoder_cfg['lr']}")

        # 3. unet
        if finetune_cfg['unet']['enable_tuning'] and finetune_cfg['unet'].get('lora_cfg'):
            unet_cfg = finetune_cfg['unet']

            where = unet_cfg['lora_cfg'].pop('where')
            assert where in ['Transformer2DModel', 'Attention']

            self.unet_lora = nn.ModuleList()
            params_list = []

            for name, module in self.unet.named_modules():
                if module.__class__.__name__ == where:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == 'Linear' or (child_module.__class__.__name__ == 'Conv2d' and child_module.kernel_size == (1, 1)):
                            lora_module = LoRALinearLayer(name + '.' + child_name, child_module, **unet_cfg['lora_cfg'])
                            self.unet_lora.append(lora_module)
                            params_list.extend(list(lora_module.parameters()))

            params_group_list.append({'params': params_list, 'lr': unet_cfg['lr']})
            logger.info(f"optimizing unet ({len(self.unet_lora)} LoRAs), using lr: {unet_cfg['lr']}")

        # 4. optimize params
        self.params_to_optimize_iterator = params_group_list

    def get_params_to_optimize(self):
        return self.params_to_optimize_iterator

    def init_new_concept(self, new_concept_tokens, initializer_tokens, enable_edlora=True):
        logger = get_logger('mixofshow', log_level='INFO')
        new_concept_cfg = {}
        new_concept_tokens = new_concept_tokens.split('+')

        if initializer_tokens is None:
            initializer_tokens = ['<rand-0.017>'] * len(new_concept_tokens)
        else:
            initializer_tokens = initializer_tokens.split('+')
        assert len(new_concept_tokens) == len(initializer_tokens), 'concept token should match init token.'

        # Initialize replace_mapping if provided in the dataset config
        if hasattr(self, 'replace_mapping'):
            new_concept_cfg['replace_mapping'] = self.replace_mapping

        for idx, (concept_name, init_token) in enumerate(zip(new_concept_tokens, initializer_tokens)):
            if enable_edlora:
                num_new_embedding = 16
            else:
                num_new_embedding = 1
            new_token_names = [f'<new{idx * num_new_embedding + layer_id}>' for layer_id in range(num_new_embedding)]
            
            # Check which tokens need to be added
            tokens_to_add = []
            token_ids = []
            for token_name in new_token_names:
                if token_name not in self.tokenizer.get_vocab():
                    tokens_to_add.append(token_name)
                token_ids.append(self.tokenizer.convert_tokens_to_ids(token_name))
            
            # Only add tokens that don't exist
            if tokens_to_add:
                num_added_tokens = self.tokenizer.add_tokens(tokens_to_add)
                assert num_added_tokens == len(tokens_to_add), 'Failed to add tokens'

            # Ensure the dictionary has the correct keys
            new_concept_cfg[concept_name] = {
                'concept_token_names': new_token_names,  # Explicitly use 'concept_token_names'
                'concept_token_ids': token_ids
            }

            # init embedding
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data

            if init_token.startswith('<rand'):
                sigma_val = float(re.findall(r'<rand-(.*)>', init_token)[0])
                init_feature = torch.randn_like(token_embeds[0]) * sigma_val
                logger.info(f'{concept_name} ({min(token_ids)}-{max(token_ids)}) is random initialized by: {init_token}')
            else:
                # Convert the initializer_token, placeholder_token to ids
                init_token_ids = self.tokenizer.encode(init_token, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(init_token_ids) > 1 or init_token_ids[0] == 40497:
                    raise ValueError('The initializer token must be a single existing token.')
                init_feature = token_embeds[init_token_ids]
                logger.info(f'{concept_name} ({min(token_ids)}-{max(token_ids)}) is random initialized by existing token ({init_token}): {init_token_ids[0]}')

            for token_id in token_ids:
                token_embeds[token_id] = init_feature.clone()

        return new_concept_cfg

    def get_all_concept_token_ids(self):
        new_concept_token_ids = []
        for key, new_token_cfg in self.new_concept_cfg.items():
            if key != 'replace_mapping':  # Skip the replace_mapping entry
                new_concept_token_ids.extend(new_token_cfg['concept_token_ids'])
        return new_concept_token_ids

    def forward(self, images, prompts, masks, img_masks):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        if self.noise_offset is not None:
            noise = noise + self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device
            )

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        ).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if self.enable_edlora:
            # Process prompts for ED-LoRA
            if isinstance(prompts, list):
                processed_prompts = []
                for prompt in prompts:
                    if '<TOK1>' in prompt or '<TOK2>' in prompt:
                        for concept_name, concept_cfg in self.new_concept_cfg.items():
                            if concept_name != 'replace_mapping':
                                if '<TOK1>' in prompt:
                                    prompt = prompt.replace('<TOK1>', concept_name)
                                    break
                        for concept_name, concept_cfg in self.new_concept_cfg.items():
                            if concept_name != 'replace_mapping':
                                if '<TOK2>' in prompt:
                                    prompt = prompt.replace('<TOK2>', concept_name)
                                    break
                    processed_prompts.append(prompt)
                prompts = processed_prompts
            prompts = bind_concept_prompt(prompts, new_concept_cfg=self.new_concept_cfg)

        # Get text embeddings
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(latents.device)
        
        # Get text encoder features
        text_encoder_output = self.text_encoder(text_inputs.input_ids)[0]

        # Get encoder hidden states and predict noise
        encoder_hidden_states = text_encoder_output
        
        if self.enable_edlora:
            encoder_hidden_states = rearrange(encoder_hidden_states, '(b n) m c -> b n m c', b=latents.shape[0])

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample

        # Calculate main loss
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)

        if self.use_mask_loss:
            loss_mask = masks
        else:
            loss_mask = img_masks

        # Calculate MSE loss with masks
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        
        # Apply mask and calculate mean
        masked_loss = torch.mul(loss, loss_mask)
        masked_loss_sum = torch.sum(masked_loss, dim=[1, 2, 3])
        mask_sum = torch.sum(loss_mask, dim=[1, 2, 3])
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        mask_sum = torch.clamp(mask_sum, min=eps)
        
        loss = torch.div(masked_loss_sum, mask_sum)
        loss = torch.mean(loss)

        # Calculate attention regularization loss if needed
        if self.attn_reg_weight is not None:
            attention_maps = self.controller.step_store
            attention_loss = self.cal_attn_reg(attention_maps, masks, text_inputs.input_ids)
            if isinstance(attention_loss, (float, int)):
                attention_loss = torch.tensor(attention_loss, device=loss.device, dtype=loss.dtype)
            if attention_loss != 0:
                loss = torch.add(loss, attention_loss)
            self.controller.reset()

        return loss

    def cal_attn_reg(self, attention_maps, masks, text_input_ids):
        '''
        attention_maps: {down_cross:[], mid_cross:[], up_cross:[]}
        masks: torch.Size([1, 1, 64, 64]) for single concept, dict for joint training
        text_input_ids: torch.Size([16, 77])
        '''
        # step 1: find token position
        batch_size = masks.shape[0] if not isinstance(masks, dict) else len(masks)
        text_input_ids = rearrange(text_input_ids, '(b l) n -> b l n', b=batch_size)
        
        # Get device from masks (either directly or from dictionary)
        device = masks.device if not isinstance(masks, dict) else next(iter(masks.values())).device
        
        new_token_pos = []
        all_concept_token_ids = self.get_all_concept_token_ids()
        
        # For single concept case, convert mask to proper format
        if not isinstance(masks, dict):
            # Check if masks are all zeros before proceeding
            if torch.all(masks == 0):
                return torch.tensor(0.0, device=device)
            
            # Convert single mask to dict format with two entries (for subject and adjective)
            masks = {
                'subject': masks.squeeze(1).clone(),  # Remove channel dimension and clone
                'adjective': masks.squeeze(1).clone()  # Same mask for both
            }
            
        for text in text_input_ids:
            # For each batch, find positions of concept tokens in each layer
            layer_positions = []
            for layer_idx, layer_text in enumerate(text):  # Iterate through each layer's text
                layer_pos = [idx for idx in range(len(layer_text)) if layer_text[idx] in all_concept_token_ids]
                if layer_pos:  # If we found concept tokens in this layer
                    layer_positions = layer_pos
                    break  # Use the first layer where we find concept tokens
            
            if not layer_positions:  # If no concept tokens found in any layer
                # Instead of defaulting to [0, 1], return 0 loss
                return torch.tensor(0.0, device=device)
            
            # For single concept case, ensure we have at least two positions
            if len(layer_positions) == 1:
                layer_positions = [layer_positions[0], layer_positions[0]]  # Use same position twice
            elif len(layer_positions) > 2:
                layer_positions = layer_positions[:2]  # Take first two positions
                
            new_token_pos.append(layer_positions)
            
        # step2: aggregate attention maps with resolution and concat heads
        attention_groups = {'64': [], '32': [], '16': [], '8': []}
        for attn_name, attention_list in attention_maps.items():
            for attn_idx, attn in enumerate(attention_list):
                if isinstance(attn, list):
                    # Skip if attention map is a list (this can happen with certain attention types)
                    continue
                res = int(math.sqrt(attn.shape[1]))
                cross_map = attn.reshape(batch_size, -1, res, res, attn.shape[-1])
                attention_groups[str(res)].append(cross_map.clone())  # Clone to avoid in-place modifications
        
        # Process each resolution group
        processed_groups = {}
        for k, maps in attention_groups.items():
            if not maps:  # Skip if no maps at this resolution
                continue
            try:
                cross_map = torch.cat(maps, dim=-4)  # concat heads
                cross_map = torch.div(torch.sum(cross_map, dim=-4), cross_map.shape[-4])  # average across heads
                cross_map = torch.stack([batch_map[..., batch_pos].clone() for batch_pos, batch_map in zip(new_token_pos, cross_map)])
                processed_groups[k] = cross_map
            except Exception as e:
                continue
            
        attn_reg_total = torch.tensor(0.0, device=device)
        # step3: calculate loss for each resolution
        for k, cross_map in processed_groups.items():
            if cross_map.shape[-1] >= 2:  # Only process if we have at least 2 attention maps
                map_adjective = cross_map[..., 0].clone()
                map_subject = cross_map[..., 1].clone()
                
                # Normalize maps with epsilon to prevent division by zero
                eps = 1e-6
                map_subject_max = torch.clamp(map_subject.max(), min=eps)
                map_adjective_max = torch.clamp(map_adjective.max(), min=eps)
                map_subject = torch.div(map_subject, map_subject_max)
                map_adjective = torch.div(map_adjective, map_adjective_max)
                
                # For single concept case, use the same mask for both maps
                if isinstance(masks, dict) and len(masks) == 1:
                    gt_mask = F.interpolate(masks['subject'].unsqueeze(1), size=map_subject.shape[1:], mode='nearest').squeeze(1)
                else:
                    gt_mask = F.interpolate(masks['subject'].unsqueeze(1), size=map_subject.shape[1:], mode='nearest').squeeze(1)
                
                # Add epsilon to denominators to prevent division by zero
                if self.reg_full_identity:
                    loss_subject = F.mse_loss(map_subject.float(), gt_mask.float(), reduction='mean')
                else:
                    non_mask_pixels = torch.clamp(torch.sum((gt_mask == 0).float()), min=eps)
                    loss_subject = torch.div(torch.sum(map_subject * (gt_mask == 0).float()), non_mask_pixels)
                
                non_mask_pixels = torch.clamp(torch.sum((gt_mask == 0).float()), min=eps)
                loss_adjective = torch.div(torch.sum(map_adjective * (gt_mask == 0).float()), non_mask_pixels)
                
                attn_reg_total = torch.add(attn_reg_total, torch.mul(self.attn_reg_weight, loss_subject + loss_adjective))
        
        return attn_reg_total

    def load_delta_state_dict(self, delta_state_dict):
        # load embedding
        logger = get_logger('mixofshow', log_level='INFO')

        if 'new_concept_embedding' in delta_state_dict and len(delta_state_dict['new_concept_embedding']) != 0:
            new_concept_tokens = list(delta_state_dict['new_concept_embedding'].keys())

            # check whether new concept is initialized
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            if set(new_concept_tokens) != set(self.new_concept_cfg.keys()):
                logger.warning('Your checkpoint have different concept with your model, loading existing concepts')

            for concept_name, concept_cfg in self.new_concept_cfg.items():
                logger.info(f'load: concept_{concept_name}')
                token_embeds[concept_cfg['concept_token_ids']] = token_embeds[
                    concept_cfg['concept_token_ids']].copy_(delta_state_dict['new_concept_embedding'][concept_name])

        # load text_encoder
        if 'text_encoder' in delta_state_dict and len(delta_state_dict['text_encoder']) != 0:
            load_keys = delta_state_dict['text_encoder'].keys()
            if hasattr(self, 'text_encoder_lora') and len(load_keys) == 2 * len(self.text_encoder_lora):
                logger.info('loading LoRA for text encoder:')
                for lora_module in self.text_encoder_lora:
                    for name, param, in lora_module.named_parameters():
                        logger.info(f'load: {lora_module.name}.{name}')
                        param.data.copy_(delta_state_dict['text_encoder'][f'{lora_module.name}.{name}'])
            else:
                for name, param, in self.text_encoder.named_parameters():
                    if name in load_keys and 'token_embedding' not in name:
                        logger.info(f'load: {name}')
                        param.data.copy_(delta_state_dict['text_encoder'][f'{name}'])

        # load unet
        if 'unet' in delta_state_dict and len(delta_state_dict['unet']) != 0:
            load_keys = delta_state_dict['unet'].keys()
            if hasattr(self, 'unet_lora') and len(load_keys) == 2 * len(self.unet_lora):
                logger.info('loading LoRA for unet:')
                for lora_module in self.unet_lora:
                    for name, param, in lora_module.named_parameters():
                        logger.info(f'load: {lora_module.name}.{name}')
                        param.data.copy_(delta_state_dict['unet'][f'{lora_module.name}.{name}'])
            else:
                for name, param, in self.unet.named_parameters():
                    if name in load_keys:
                        logger.info(f'load: {name}')
                        param.data.copy_(delta_state_dict['unet'][f'{name}'])

    def delta_state_dict(self):
        delta_dict = {'new_concept_embedding': {}, 'text_encoder': {}, 'unet': {}}

        # save_embedding
        for concept_name, concept_cfg in self.new_concept_cfg.items():
            if concept_name != 'replace_mapping':  # Skip the replace_mapping entry
                learned_embeds = self.text_encoder.get_input_embeddings().weight[concept_cfg['concept_token_ids']]
                delta_dict['new_concept_embedding'][concept_name] = learned_embeds.detach().cpu()

        # save text model
        if hasattr(self, 'text_encoder_lora'):
            for lora_module in self.text_encoder_lora:
                for name, param in lora_module.named_parameters():
                    delta_dict['text_encoder'][f'{lora_module.name}.{name}'] = param.detach().cpu()

        # save unet
        if hasattr(self, 'unet_lora'):
            for lora_module in self.unet_lora:
                for name, param in lora_module.named_parameters():
                    delta_dict['unet'][f'{lora_module.name}.{name}'] = param.detach().cpu()