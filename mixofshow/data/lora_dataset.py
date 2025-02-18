import json
import os
import random
import re
from pathlib import Path
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
import torch
from colorama import Fore

from mixofshow.data.pil_transform import PairCompose, build_transforms


class LoraDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.instance_prompt = opt.get('instance_prompt', '')
        self.instance_data_dir = opt.get('instance_data_dir', None)
        self.caption_dir = opt.get('caption_dir', None)
        self.mask_dir = opt.get('mask_dir', None)
        self.use_caption = opt.get('use_caption', False)
        self.use_mask = opt.get('use_mask', False)
        
        # Check if this is joint training by looking for subdirectories in mask_dir
        if self.mask_dir and os.path.exists(self.mask_dir):
            subdirs = [d for d in os.listdir(self.mask_dir) if os.path.isdir(os.path.join(self.mask_dir, d))]
            self.is_joint_training = len(subdirs) > 0
            if self.is_joint_training:
                # Load concept list to get mask_tokens mapping
                if 'concept_list' in opt:
                    with open(opt['concept_list'], 'r') as f:
                        concept_list = json.load(f)
                        if len(concept_list) > 0 and 'mask_tokens' in concept_list[0]:
                            self.mask_tokens = concept_list[0]['mask_tokens']
                        else:
                            self.mask_tokens = {}
                else:
                    self.mask_tokens = {}
        else:
            self.is_joint_training = False
        
        # Get image paths with explicit pattern
        if self.instance_data_dir is not None:
            image_pattern = os.path.join(self.instance_data_dir, '*') 
            self.image_paths = sorted(glob.glob(image_pattern))
            
            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {self.instance_data_dir}")
            
            self.num_instance_images = len(self.image_paths)
            
            # Set the length based on dataset_enlarge_ratio
            enlarge_ratio = self.opt.get('dataset_enlarge_ratio', 1)
            self.length = self.num_instance_images * int(enlarge_ratio)
        else:
            raise ValueError("instance_data_dir must be provided")
        
        # Set up transforms
        if 'instance_transform' in opt:
            self.transform = build_transforms(opt['instance_transform'])
        else:
            self.transform = None

    def process_text(self, instance_prompt, replace_mapping):
        for k, v in replace_mapping.items():
            instance_prompt = instance_prompt.replace(k, v)
        instance_prompt = instance_prompt.strip()
        instance_prompt = re.sub(' +', ' ', instance_prompt)
        return instance_prompt

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get image path and load image
        real_idx = idx % self.num_instance_images  # Handle dataset enlargement
        image_path = self.image_paths[real_idx]
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
        image = Image.open(image_path).convert('RGB')
        
        # Get caption
        if self.use_caption and self.caption_dir is not None:
            caption_path = os.path.join(self.caption_dir, base_name + '.txt')
            with open(caption_path, 'r') as f:
                prompt = f.read().strip()
        else:
            prompt = self.instance_prompt

        # Create kwargs dictionary with prompt
        kwargs = {'prompts': prompt}
        
        # Handle mask loading
        if self.use_mask and self.mask_dir is not None:
            if self.is_joint_training:  # Joint training case
                masks = {}
                # First try to use mask_tokens mapping
                if hasattr(self, 'mask_tokens') and self.mask_tokens:
                    for token, subdir in self.mask_tokens.items():
                        mask_path = os.path.join(self.mask_dir, subdir, image_name)
                        if os.path.exists(mask_path):
                            try:
                                mask = Image.open(mask_path).convert('L')
                                mask_tensor = ToTensor()(mask)
                                mask_tensor = (mask_tensor > 0.5).float()
                                
                                if mask_tensor.sum() == 0:
                                    continue
                                    
                                mask_tensor = torch.nn.functional.interpolate(
                                    mask_tensor.unsqueeze(0), 
                                    size=(64, 64), 
                                    mode='nearest'
                                ).squeeze(0)
                                
                                if mask_tensor.sum() == 0:
                                    continue
                                    
                                masks[token] = mask_tensor
                            except (IOError, OSError):
                                continue
                # Fallback to mask_mapping if no masks were loaded
                if not masks and 'mask_mapping' in self.opt and base_name in self.opt['mask_mapping']:
                    mapping = self.opt['mask_mapping'][base_name]
                    for token, mask_path in mapping.items():
                        full_mask_path = os.path.join(self.mask_dir, mask_path)
                        if os.path.exists(full_mask_path):
                            try:
                                mask = Image.open(full_mask_path).convert('L')
                                mask_tensor = ToTensor()(mask)
                                mask_tensor = (mask_tensor > 0.5).float()
                                
                                if mask_tensor.sum() == 0:
                                    continue
                                    
                                mask_tensor = torch.nn.functional.interpolate(
                                    mask_tensor.unsqueeze(0), 
                                    size=(64, 64), 
                                    mode='nearest'
                                ).squeeze(0)
                                
                                if mask_tensor.sum() == 0:
                                    continue
                                    
                                masks[token] = mask_tensor
                            except (IOError, OSError):
                                continue
                if masks:  # Only add masks if we found any valid ones
                    kwargs['mask'] = masks
            else:  # Single concept case
                mask_path = os.path.join(self.mask_dir, base_name + '.png')
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path).convert('L')
                        mask_tensor = ToTensor()(mask)
                        mask_tensor = (mask_tensor > 0.5).float()
                        
                        if mask_tensor.sum() == 0:
                            return None
                            
                        mask_tensor = torch.nn.functional.interpolate(
                            mask_tensor.unsqueeze(0), 
                            size=(64, 64), 
                            mode='nearest'
                        )  # Keep the batch dimension for single concept case
                        
                        if mask_tensor.sum() == 0:
                            return None
                            
                        if mask_tensor.max() > 0:
                            mask_tensor = mask_tensor / mask_tensor.max()
                            
                        kwargs['mask'] = mask_tensor
                    except (IOError, OSError):
                        pass
        
        # Apply transforms to image with kwargs
        if self.transform is not None:
            transformed = self.transform(image, **kwargs)
            if isinstance(transformed, tuple):
                image = transformed[0]  # Take first element if tuple
                kwargs = transformed[1]  # Update kwargs with transformed values
        
        # Create empty mask tensor if no masks were found
        if 'mask' not in kwargs:
            if self.is_joint_training:
                # For joint training, create empty dict with empty tensors
                empty_masks = {}
                if hasattr(self, 'mask_tokens'):
                    for token in self.mask_tokens:
                        empty_masks[token] = torch.zeros((1, image.shape[1] // 8, image.shape[2] // 8))
                masks = empty_masks
            else:
                # For single concept, create single empty tensor with batch dimension
                masks = torch.zeros((1, 1, image.shape[1] // 8, image.shape[2] // 8))
        else:
            masks = kwargs['mask']
        
        return {
            'image_paths': image_path,
            'images': image,
            'prompts': kwargs.get('prompts', prompt),  # Use transformed prompt if available
            'masks': masks
        }
