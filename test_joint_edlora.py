import argparse
import os
import os.path as osp
import datetime
import glob
import hashlib
import json
import re

import torch
import torch.utils.checkpoint
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image, ImageDraw
from colorama import Fore
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline
from mixofshow.pipelines.pipeline_regionally_t2iadapter import RegionallyT2IAdapterPipeline
from mixofshow.utils.convert_edlora_to_diffusers import convert_edlora
from mixofshow.utils.util import NEGATIVE_PROMPT, compose_visualize, dict2str, pil_imwrite, set_path_logger

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version('0.18.2')

def find_latest_model(base_dir='experiments', enable_edlora=True):
    """Find the most recently saved model across all archived directories."""
    # Pattern to match both main and archived directories
    exp_pattern = os.path.join(base_dir, 'EDLoRA_cat2_dog6_joint*')
    exp_dirs = glob.glob(exp_pattern)
    
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found matching {exp_pattern}")
    
    latest_model = None
    latest_time = None
    lora_type = 'edlora' if enable_edlora else 'lora'
    
    for exp_dir in exp_dirs:
        model_dir = os.path.join(exp_dir, 'models')
        if not os.path.exists(model_dir):
            continue
            
        # Check all model files in this directory
        model_pattern = os.path.join(model_dir, f'{lora_type}_model-*.pth')
        model_files = glob.glob(model_pattern)
        
        for model_file in model_files:
            model_time = os.path.getmtime(model_file)
            if latest_time is None or model_time > latest_time:
                latest_time = model_time
                latest_model = model_file
    
    if latest_model is None:
        raise FileNotFoundError(f"No model files found in any experiment directory")
        
    # Convert timestamp to readable format
    latest_time_str = datetime.datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{Fore.GREEN}Found most recent model:{Fore.RESET}")
    print(f"{Fore.GREEN}  Path: {latest_model}{Fore.RESET}")
    print(f"{Fore.GREEN}  Last modified: {latest_time_str}{Fore.RESET}")
    
    return latest_model

def generate_regional_prompts(prompt, token_pairs, context_neg_prompt, prompt_mask_dir, prompt_idx, scale_x=1):
    """
    Generate regional prompts with mask-based bounding boxes.
    
    Args:
        prompt (str): Original prompt
        token_pairs (list): List of token pairs
        context_neg_prompt (str): Negative prompt for the context
        prompt_mask_dir (str): Directory containing masks
        prompt_idx (int): Prompt index for logging
        scale_x (int, optional): Scale factor for x-coordinates
    
    Returns:
        tuple: (context_prompt, region_prompts, prompt_rewrite)
    """
    # Use the original prompt as the context prompt
    context_prompt = prompt
    
    # Target dimensions (what the coordinates will be normalized against)
    target_height = 512
    target_width = 512  # Keep this at 512, we'll handle scaling separately
    
    # Prepare region prompts and bounding boxes
    region_prompts = []
    
    for first_token, second_token in token_pairs:
        # Clean tokens
        first_token_clean = first_token[1:-1]  # Remove < >
        second_token_clean = second_token[1:-1]  # Remove < >
        
        # Construct mask path
        mask_path = os.path.join(prompt_mask_dir, f"{first_token_clean}_{second_token_clean}.jpg")
        
        # Try to get mask-based bounding box
        if os.path.exists(mask_path):
            # Save bounding box mask
            bbox_mask_path = os.path.join(prompt_mask_dir, f"{first_token_clean}_{second_token_clean}_bbox.jpg")
            min_y, min_x, max_y, max_x = save_bounding_box_mask(mask_path, bbox_mask_path)
            
            # First normalize to 0-1 range based on original dimensions
            min_y = min_y / target_height
            max_y = max_y / target_height
            min_x = min_x / target_width
            max_x = max_x / target_width
            
            # Then scale x coordinates, keeping them in 0-1 range
            min_x = min_x / scale_x
            max_x = max_x / scale_x
            
            region = [min_y, min_x, max_y, max_x]
            print(f"{Fore.CYAN}Region for {first_token_clean} {second_token_clean}:{Fore.RESET}")
            print(f"{Fore.CYAN}  Original bbox (pixels): y=[{min_y * target_height}, {max_y * target_height}], x=[{min_x * target_width * scale_x}, {max_x * target_width * scale_x}]{Fore.RESET}")
            print(f"{Fore.CYAN}  Normalized coords: y=[{min_y}, {max_y}], x=[{min_x}, {max_x}]{Fore.RESET}")
        else:
            print(f"{Fore.YELLOW}Warning: No mask found for {first_token_clean} {second_token_clean}{Fore.RESET}")
            region = [0, 0, 1, 1]  # Use normalized coordinates for default region
            print(f"{Fore.YELLOW}Using default region: {region}{Fore.RESET}")
        
        # Construct region prompt
        region_prompt = f"[a {first_token} {second_token}, in the scene]"
        region_neg_prompt = f"[{context_neg_prompt}]"
        
        # Create full region specification
        full_region_prompt = f"{region_prompt}-*-{region_neg_prompt}-*-{region}"
        region_prompts.append(full_region_prompt)
    
    # Join region prompts
    prompt_rewrite = '|'.join(region_prompts)
    
    # Debug print
    print(f"{Fore.CYAN}Context Prompt: {context_prompt}{Fore.RESET}")
    print(f"{Fore.CYAN}Prompt Rewrite: {prompt_rewrite}{Fore.RESET}")
    
    return context_prompt, region_prompts, prompt_rewrite

def generate_regional_script(prompt, region_prompts, opt, prompt_idx, spatial_dir, prompt_mask_dir, lora_path, context_prompt):
    """Generate content for a custom regionally_sample.sh script."""
    
    # Get absolute paths
    abs_spatial_dir = os.path.abspath(spatial_dir)
    abs_prompt_mask_dir = os.path.abspath(prompt_mask_dir)
    
    # Find the latest combined model directory
    model_dir = os.path.dirname(lora_path)
    combined_model_dirs = glob.glob(os.path.join(model_dir, 'combined_model*'))
    if not combined_model_dirs:
        # If no combined_model directories found, look for checkpoint directories
        combined_model_dirs = glob.glob(os.path.join(model_dir, 'checkpoint-*', 'combined_model*'))
    
    if not combined_model_dirs:
        raise ValueError(f"No combined model directories found in {model_dir}")
    
    # Extract step numbers and find the largest one
    step_numbers = []
    for dir_path in combined_model_dirs:
        try:
            # Try to extract step number from the directory path
            match = re.search(r'(?:checkpoint-)?(\d+)', dir_path)
            if match:
                step = int(match.group(1))
                step_numbers.append((step, dir_path))
        except ValueError:
            continue
    
    if not step_numbers:
        raise ValueError(f"No valid step numbers found in combined model directories")
    
    # Sort and get the latest step
    latest_step, latest_model_path = max(step_numbers, key=lambda x: x[0])
    print(f"Using latest model at step {latest_step}: {latest_model_path}")

    abs_model_path = os.path.abspath(latest_model_path)

    # Base configuration
    script_content = [
        "#!/bin/bash",
        "",
        "# Generated regional sampling script",
        f"fused_model=\"{abs_model_path}\"",
        "expdir=\"custom_regional_sample\"",
        "",
        "# Set default weights",
        "keypose_condition=''",
        "keypose_adaptor_weight=0.0",  # Set to 0 since we're not using it
        f"sketch_condition=\"{os.path.join(abs_spatial_dir, str(prompt_idx), f'spatial_mask_{prompt_idx}.jpg')}\"",
        "sketch_adaptor_weight=1.0",
        "",
        f"context_prompt='{context_prompt}'",
        f"context_neg_prompt='{NEGATIVE_PROMPT}'",
        ""
    ]
    
    # Add region variables
    for i, region_prompt in enumerate(region_prompts, 1):
        prompt_part, neg_prompt_part, box_part = region_prompt.split('-*-')
        script_content.extend([
            f"region{i}_prompt='{prompt_part}'",
            f"region{i}_neg_prompt='{neg_prompt_part}'",
            f"region{i}='{box_part}'",
            ""
        ])
    
    # Build prompt_rewrite using region variables
    region_vars = []
    for i in range(1, len(region_prompts) + 1):
        region_vars.append(f"${{region{i}_prompt}}-*-${{region{i}_neg_prompt}}-*-${{region{i}}}")
    
    prompt_rewrite = '|'.join(region_vars)
    
    # Add the rest of the script
    script_content.extend([
        f"prompt_rewrite=\"{prompt_rewrite}\"",
        "",
        "python regionally_controlable_sampling.py \\",
        "  --pretrained_model=${fused_model} \\",
        "  --sketch_adaptor_weight=${sketch_adaptor_weight} \\",
        "  --sketch_condition=${sketch_condition} \\",
        "  --keypose_adaptor_weight=${keypose_adaptor_weight} \\",
        "  --keypose_condition=${keypose_condition} \\",
        f"  --save_dir=\"{abs_prompt_mask_dir}\" \\",
        "  --prompt=\"${context_prompt}\" \\",
        "  --negative_prompt=\"${context_neg_prompt}\" \\",
        "  --prompt_rewrite=\"${prompt_rewrite}\" \\",
        "  --suffix=\"baseline\" \\",
        "  --seed=42"
    ])
    
    return '\n'.join(script_content)

def create_hollow_mask(mask_path, thickness=5):
    """Create a hollow edge mask from a filled mask."""
    # Read the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty image for the hollow mask
    hollow_mask = np.zeros_like(mask)
    
    # Draw contours with specified thickness
    cv2.drawContours(hollow_mask, contours, -1, (255), thickness)
    
    return hollow_mask

def calculate_mask_overlap(mask1_path, mask2_path):
    """
    Calculate the overlap percentage between two masks.
    
    Args:
        mask1_path (str): Path to the first mask image
        mask2_path (str): Path to the second mask image
    
    Returns:
        float: Percentage of overlap between the two masks
    """
    # Read masks
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure masks are the same size
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    
    # Calculate overlap percentage
    intersection_pixels = np.sum(intersection > 0)
    union_pixels = np.sum(union > 0)
    
    # Avoid division by zero
    if union_pixels == 0:
        return 0.0
    
    overlap_percentage = (intersection_pixels / union_pixels) * 100
    
    return overlap_percentage

def resample_image(base_pipe, prompt, negative_prompt=NEGATIVE_PROMPT, seed=None):
    """
    Resample an image with optional seed for reproducibility.
    
    Args:
        base_pipe (StableDiffusionPipeline): The pipeline to use for generation
        prompt (str): Prompt for image generation
        negative_prompt (str, optional): Negative prompt. Defaults to NEGATIVE_PROMPT.
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        PIL.Image: Regenerated image
    """
    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device='cuda:1').manual_seed(seed)
    else:
        generator = None
    
    # Generate image
    output = base_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512,
    ).images[0]
    
    return output

def visual_validation_simple(accelerator, pipe, dataloader, current_iter, opt):
    dataset_name = dataloader.dataset.opt['name']
    pipe.unet.eval()
    pipe.text_encoder.eval()

    for idx, val_data in enumerate(tqdm(dataloader)):
        output = pipe(
            prompt=val_data['prompts'],
            latents=val_data['latents'].to(device='cuda:1', dtype=torch.float16),
            negative_prompt=[NEGATIVE_PROMPT] * len(val_data['prompts']),
            num_inference_steps=opt['val']['sample'].get('num_inference_steps', 50),
            guidance_scale=opt['val']['sample'].get('guidance_scale', 7.5),
        ).images

        for img, prompt, indice in zip(output, val_data['prompts'], val_data['indices']):
            img_name = '{prompt}---G_{guidance_scale}_S_{steps}---{indice}'.format(
                prompt=prompt.replace(' ', '_'),
                guidance_scale=opt['val']['sample'].get('guidance_scale', 7.5),
                steps=opt['val']['sample'].get('num_inference_steps', 50),
                indice=indice)

            save_img_path = osp.join(opt['path']['visualization'], dataset_name, f'{current_iter}', f'{img_name}---{current_iter}.png')

            pil_imwrite(img, save_img_path)
            print(f"Saved image to {save_img_path}")
        # tentative for out of GPU memory
        del output
        torch.cuda.empty_cache()

    # Save the lora layers, final eval
    accelerator.wait_for_everyone()

    if opt['val'].get('compose_visualize'):
        if accelerator.is_main_process:
            compose_visualize(os.path.dirname(save_img_path))

def visual_validation(accelerator, base_pipe, dataloader, current_iter, opt, test_dir, lora_path):
    dataset_name = dataloader.dataset.opt['name']
    base_pipe.unet.eval()
    base_pipe.text_encoder.eval()

    # Create dataset-specific directory
    dataset_dir = os.path.join(test_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Create iteration-specific directory
    iter_dir = os.path.join(dataset_dir, current_iter)
    os.makedirs(iter_dir, exist_ok=True)

    # Create base mask directory
    base_mask_dir = os.path.join(iter_dir, 'masks')
    os.makedirs(base_mask_dir, exist_ok=True)
    print(f"{Fore.YELLOW}Base mask directory: {base_mask_dir}{Fore.RESET}")

    # Create spatial adaptor directory
    spatial_dir = os.path.join(dataset_dir, 'spatial_adaptor')
    os.makedirs(spatial_dir, exist_ok=True)
    print(f"{Fore.YELLOW}Spatial adaptor directory: {spatial_dir}{Fore.RESET}")

    for idx, val_data in enumerate(tqdm(dataloader)):
        # Move all tensors to the same device as the pipe's unet
        device = 'cuda:1'
        val_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_data.items()}

        # First generate image without region control
        initial_output = base_pipe(
            prompt=val_data['prompts'],
            negative_prompt=[NEGATIVE_PROMPT] * len(val_data['prompts']),
            num_inference_steps=opt['val']['sample'].get('num_inference_steps', 50),
            guidance_scale=opt['val']['sample'].get('guidance_scale', 7.5),
            height=512,  # Default size
            width=512,   # Default size
        ).images

        print(f"{Fore.CYAN}Using region control for generation{Fore.RESET}")
        for prompt_idx, (prompt, initial_image) in enumerate(zip(val_data['prompts'], initial_output)):
            # Create prompt-specific directories
            prompt_mask_dir = os.path.join(base_mask_dir, str(prompt_idx))
            prompt_spatial_dir = os.path.join(spatial_dir, str(prompt_idx))
            os.makedirs(prompt_mask_dir, exist_ok=True)
            os.makedirs(prompt_spatial_dir, exist_ok=True)

            # Extract special tokens
            tokens = [obj.strip() for obj in prompt.split() if obj.strip().startswith('<') and obj.strip().endswith('>')]
            
            # Group tokens into pairs
            token_pairs = []
            for i in range(0, len(tokens), 2):
                if i + 1 < len(tokens):  # Make sure we have a pair
                    token_pair = (tokens[i], tokens[i+1])
                    token_pairs.append(token_pair)
            
            if not token_pairs or len(token_pairs) < 2:
                continue
            
            # Construct context negative prompt
            concept_neg_prompts = {
                'cat': "deformed, ugly, poor quality, low resolution, blurry, cartoon-like, unrealistic cat, cat with human features, mutated cat",
                'dog': "deformed, ugly, poor quality, low resolution, blurry, cartoon-like, unrealistic dog, dog with human features, mutated dog",
                'default': "deformed, ugly, poor quality, low resolution, blurry, cartoon-like"
            }
            
            # Combine concept-specific negative prompts
            context_neg_prompt = f"{NEGATIVE_PROMPT}"

            # Save initial image for segmentation
            temp_image_path = os.path.join(prompt_mask_dir, f'temp_{prompt_idx}.jpg')
            initial_image.save(temp_image_path)

            # Create segmentation text using both tokens of each pair
            text_conditions = []
            for first_token, second_token in token_pairs:
                # Remove < > from both tokens and combine them
                combined_tokens = f"{first_token[1:-1]} {second_token[1:-1]}"
                text_conditions.append(combined_tokens)
            
            text_condition = '+'.join(text_conditions)

            # Run text-guided segmentation
            seg_cmd = f'CUDA_VISIBLE_DEVICES=1 python text_segment/run_expand.py --input_path={temp_image_path} --text_condition="{text_condition}" --output_path={prompt_mask_dir} --prompt_idx={prompt_idx} --sam_checkpoint=./sam_vit_h_4b8939.pth'
            
            # Clear CUDA cache before running segmentation
            torch.cuda.empty_cache()
            
            # Run segmentation and check for success
            ret = os.system(seg_cmd)
            if ret != 0:
                continue

            # Check mask overlap
            max_overlap_percentage = 50  # Maximum allowed overlap
            max_resampling_attempts = 5
            
            # Collect mask paths
            mask_paths = []
            for first_token, second_token in token_pairs:
                # Use both tokens for the mask filename
                first_token_clean = first_token[1:-1]  # Remove < >
                second_token_clean = second_token[1:-1]  # Remove < >
                mask_path = os.path.join(prompt_mask_dir, f"{first_token_clean}_{second_token_clean}.jpg")
                
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
            
            # If we have at least 2 masks, check overlap
            if len(mask_paths) >= 2:
                # Check overlap between first two masks
                overlap = calculate_mask_overlap(mask_paths[0], mask_paths[1])
                
                # Resample if overlap is too high
                resampling_attempt = 0
                while overlap > max_overlap_percentage and resampling_attempt < max_resampling_attempts:
                    print(f"{Fore.YELLOW}Mask overlap too high: {overlap:.2f}%. Resampling...{Fore.RESET}")
                    
                    # Resample with a different seed
                    resampled_image = resample_image(base_pipe, prompt, seed=resampling_attempt)
                    
                    # Save resampled image
                    temp_image_path = os.path.join(prompt_mask_dir, f'temp_{prompt_idx}_resample_{resampling_attempt}.jpg')
                    resampled_image.save(temp_image_path)
                    
                    # Run segmentation on resampled image
                    seg_cmd = f'CUDA_VISIBLE_DEVICES=1 python text_segment/run_expand.py --input_path={temp_image_path} --text_condition="{text_condition}" --output_path={prompt_mask_dir} --prompt_idx={prompt_idx} --sam_checkpoint=./sam_vit_h_4b8939.pth'
                    
                    ret = os.system(seg_cmd)
                    if ret != 0:
                        resampling_attempt += 1
                        continue
                    
                    # Recollect mask paths
                    mask_paths = []
                    for first_token, second_token in token_pairs:
                        first_token_clean = first_token[1:-1]
                        second_token_clean = second_token[1:-1]
                        mask_path = os.path.join(prompt_mask_dir, f"{first_token_clean}_{second_token_clean}.jpg")
                        
                        if os.path.exists(mask_path):
                            mask_paths.append(mask_path)
                    
                    # Recheck overlap
                    if len(mask_paths) >= 2:
                        overlap = calculate_mask_overlap(mask_paths[0], mask_paths[1])
                    
                    resampling_attempt += 1
                
                # If we couldn't reduce overlap, skip this prompt
                if overlap > max_overlap_percentage:
                    print(f"{Fore.RED}Could not reduce mask overlap. Skipping prompt.{Fore.RESET}")
                    continue

            # After generating masks, create hollow edge masks
            combined_hollow_mask = np.zeros((512, 512), dtype=np.uint8)
            
            mask_paths = []
            for first_token, second_token in token_pairs:
                # Use both tokens for the mask filename
                first_token_clean = first_token[1:-1]  # Remove < >
                second_token_clean = second_token[1:-1]  # Remove < >
                mask_path = os.path.join(prompt_mask_dir, f"{first_token_clean}_{second_token_clean}.jpg")
                
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    
                    # Create hollow edge mask
                    hollow_mask = create_hollow_mask(mask_path)
                    # Combine with previous masks
                    combined_hollow_mask = cv2.bitwise_or(combined_hollow_mask, hollow_mask)
            
            # Scale the mask to double width using interpolation
            doubled_width_mask = cv2.resize(combined_hollow_mask, (1024, 512), interpolation=cv2.INTER_LINEAR)
            
            # Save the scaled width hollow mask
            spatial_mask_path = os.path.join(prompt_spatial_dir, f'spatial_mask_{prompt_idx}.jpg')
            cv2.imwrite(spatial_mask_path, doubled_width_mask)

            # Generate regional prompts with doubled x-coordinates
            context_prompt, region_prompts, prompt_rewrite = generate_regional_prompts(
                prompt, token_pairs, context_neg_prompt, prompt_mask_dir, prompt_idx, scale_x=2
            )

            # Generate regional sampling script
            script_content = generate_regional_script(
                prompt, region_prompts, opt, prompt_idx, 
                spatial_dir, prompt_mask_dir, lora_path, context_prompt
            )
            
            script_path = os.path.join(prompt_mask_dir, f'regional_sample_{prompt_idx}.sh')
            
            # Write script to file
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Run the script
            ret = os.system(f"bash {script_path}")
            if ret != 0:
                continue

        # Clean up GPU memory
        torch.cuda.empty_cache()

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Compose visualization if requested
    if opt['val'].get('compose_visualize') and accelerator.is_main_process:
        compose_visualize(iter_dir)

def test(root_path, args):
    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, opt['name'], opt, args.opt, is_train=False)

    # get logger
    logger = get_logger('mixofshow', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)

    logger.info(dict2str(opt))

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'])

    # Get the training dataset
    valset_cfg = opt['datasets']['val_vis']
    val_dataset = PromptDataset(valset_cfg)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=valset_cfg['batch_size_per_gpu'], shuffle=False)

    enable_edlora = opt['models']['enable_edlora']

    for lora_alpha in opt['val']['alpha_list']:
        # Initialize base pipeline for initial image generation
        base_pipeclass = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline
        base_pipe = base_pipeclass.from_pretrained(
            opt['models']['pretrained_path'],
            scheduler=DPMSolverMultistepScheduler.from_pretrained(opt['models']['pretrained_path'], subfolder='scheduler'),
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to('cuda:1')

        # Load the model
        if args.model_path:
            lora_path = args.model_path
        else:
            # Find the most recently saved model across all directories
            lora_path = find_latest_model(enable_edlora=enable_edlora)

        print(f"{Fore.YELLOW}Using model file: {lora_path}{Fore.RESET}")

        base_pipe, new_concept_cfg = convert_edlora(base_pipe, torch.load(lora_path), enable_edlora=enable_edlora, alpha=lora_alpha)
        base_pipe.set_new_concept_cfg(new_concept_cfg)

        # Initialize regional pipeline if needed
        if args.region_control:
            regional_pipe = RegionallyT2IAdapterPipeline.from_pretrained(
                opt['models']['pretrained_path'],
                scheduler=DPMSolverMultistepScheduler.from_pretrained(opt['models']['pretrained_path'], subfolder='scheduler'),
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to('cuda:1')
            regional_pipe, _ = convert_edlora(regional_pipe, torch.load(lora_path), enable_edlora=enable_edlora, alpha=lora_alpha)
            regional_pipe.set_new_concept_cfg(new_concept_cfg)
        else:
            regional_pipe = None

        # visualize embedding + LoRA weight shift
        logger.info(f'Start validation sample lora({lora_alpha}):')

        lora_type = 'edlora' if enable_edlora else 'lora'
        if args.region_control:
            visual_validation(accelerator, base_pipe, val_dataloader, 
                             f'validation_{lora_type}_{lora_alpha}', opt, test_dir, lora_path)
        else:
            visual_validation_simple(accelerator, base_pipe, val_dataloader,  f'validation_{lora_type}_{lora_alpha}', opt)
        del base_pipe
        if regional_pipe is not None:
            del regional_pipe

def save_bounding_box_mask(mask_path, output_path=None, target_size=(512, 512)):
    """
    Create and save a black and white mask representing the bounding box.
    
    Args:
        mask_path (str): Path to the input mask image
        output_path (str, optional): Path to save the bounding box mask. 
                                     If None, uses the input path with '_bbox' suffix.
        target_size (tuple, optional): Target image size to scale coordinates to. 
                                       Defaults to (512, 512)
    
    Returns:
        tuple: Scaled bounding box coordinates (ymin, xmin, ymax, xmax)
    """
    # Read the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Get original mask dimensions
    orig_h, orig_w = mask.shape
    
    # Find non-zero points
    y_indices, x_indices = np.nonzero(mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # If no non-zero points, return a blank mask and default coordinates
        bbox_mask = np.zeros(target_size, dtype=np.uint8)
        return 0, 0, target_size[0], target_size[1]
    
    # Get bounding box coordinates
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    
    # Scale coordinates to target size
    scale_y = target_size[0] / orig_h
    scale_x = target_size[1] / orig_w
    
    scaled_ymin = int(ymin * scale_y)
    scaled_ymax = int(ymax * scale_y)
    scaled_xmin = int(xmin * scale_x)
    scaled_xmax = int(xmax * scale_x)
    
    print(f"Original mask size: {orig_h}x{orig_w}")
    print(f"Original bbox: ({xmin}, {ymin}) to ({xmax}, {ymax})")
    print(f"Scaled bbox: ({scaled_xmin}, {scaled_ymin}) to ({scaled_xmax}, {scaled_ymax})")
    
    # Create a blank mask at target size
    bbox_mask = np.zeros(target_size, dtype=np.uint8)
    
    # Draw the scaled bounding box
    cv2.rectangle(bbox_mask, (scaled_xmin, scaled_ymin), (scaled_xmax, scaled_ymax), 255, -1)
    
    # If output path is not provided, generate one
    if output_path is None:
        output_path = mask_path.replace('.jpg', '_bbox.jpg')
    
    # Save the bounding box mask
    cv2.imwrite(output_path, bbox_mask)
    
    return scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/test/EDLoRA/EDLoRA_hina_Anyv4_B4_Iter1K.yml')
    parser.add_argument('-model_path', type=str, default=None, help='Path to the trained model file')
    parser.add_argument('--region_control', action='store_true', help='Enable automatic region control using Grounding DINO')
    args = parser.parse_args()

    # Create test directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join('experiments', 'test', timestamp)
    os.makedirs(test_dir, exist_ok=True)
    print(f"{Fore.YELLOW}Test results will be saved to: {test_dir}{Fore.RESET}")

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test(root_path, args)
