import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from mixofshow.pipelines.attention import AttentionStore
from mixofshow.pipelines.trainer_edlora import EDLoRATrainer
from mixofshow.pipelines.utils import bind_concept_prompt, revise_edlora_unet_attention_controller_forward, revise_edlora_unet_attention_forward


class JointEDLoRATrainer(nn.Module):
    def __init__(
        self,
        pretrained_path,
        concept_1_cfg,  # new_concept_token and initializer_token for concept 1
        concept_2_cfg,  # new_concept_token and initializer_token for concept 2
        enable_edlora,  # true for ED-LoRA, false for LoRA
        finetune_cfg=None,
        noise_offset=None,
        attn_reg_weight=None,
        reg_full_identity=True,  # True for thanos, False for real person (don't need to encode clothes)
        use_mask_loss=True,
        enable_xformers=False,
        gradient_checkpoint=False
    ):
        super().__init__()

        # Create individual trainers for each concept
        self.trainer_1 = EDLoRATrainer(
            pretrained_path=pretrained_path,
            new_concept_token=concept_1_cfg['new_concept_token'],
            initializer_token=concept_1_cfg['initializer_token'],
            enable_edlora=enable_edlora,
            finetune_cfg=finetune_cfg,
            noise_offset=noise_offset,
            attn_reg_weight=attn_reg_weight,
            reg_full_identity=reg_full_identity,
            use_mask_loss=use_mask_loss,
            enable_xformers=enable_xformers,
            gradient_checkpoint=gradient_checkpoint
        )

        self.trainer_2 = EDLoRATrainer(
            pretrained_path=pretrained_path,
            new_concept_token=concept_2_cfg['new_concept_token'],
            initializer_token=concept_2_cfg['initializer_token'],
            enable_edlora=enable_edlora,
            finetune_cfg=finetune_cfg,
            noise_offset=noise_offset,
            attn_reg_weight=attn_reg_weight,
            reg_full_identity=reg_full_identity,
            use_mask_loss=use_mask_loss,
            enable_xformers=enable_xformers,
            gradient_checkpoint=gradient_checkpoint
        )

        # Share the models between trainers
        self.trainer_2.vae = self.trainer_1.vae
        self.trainer_2.tokenizer = self.trainer_1.tokenizer
        self.trainer_2.text_encoder = self.trainer_1.text_encoder
        self.trainer_2.unet = self.trainer_1.unet
        self.trainer_2.scheduler = self.trainer_1.scheduler

        self.enable_edlora = enable_edlora
        self.use_mask_loss = use_mask_loss
        self.attn_reg_weight = attn_reg_weight

    def forward(self, images, prompts, masks, img_masks, dataset_type=None):
        """
        Forward pass that handles both individual and joint training.
        dataset_type can be 'cat', 'dog', or 'joint'
        """
        if dataset_type == 'cat':
            # Only train on cat data
            return self.trainer_1(images, prompts, masks['<TOK1>'], img_masks)
        elif dataset_type == 'dog':
            # Only train on dog data
            return self.trainer_2(images, prompts, masks['<TOK2>'], img_masks)
        else:  # joint training
            # Calculate individual losses
            loss_1 = self.trainer_1(images, prompts, masks['<TOK1>'], img_masks)
            loss_2 = self.trainer_2(images, prompts, masks['<TOK2>'], img_masks)

            # Calculate joint loss (VCG-like mechanism)
            # Each concept's loss includes its own loss plus the losses from other concepts
            joint_loss = loss_1 + loss_2

            return joint_loss

    def save_pretrained(self, save_path):
        # Save both trainers
        self.trainer_1.save_pretrained(f"{save_path}_concept1")
        self.trainer_2.save_pretrained(f"{save_path}_concept2")