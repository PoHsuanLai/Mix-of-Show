import argparse
import os
import numpy as np
from PIL import Image
from lang_sam import LangSAM
import torch
from colorama import Fore

def get_best_mask_for_concept(model, image_pil, concept, all_concepts):
    """Get the best mask for a specific concept by comparing with other concepts."""
    # Get masks for this concept
    masks, boxes, phrases, logits = model.predict(image_pil, concept)
    
    if len(masks) == 0:
        print(f"{Fore.RED}No masks found for concept: {concept}{Fore.RESET}")
        return None
        
    # Get masks for other concepts for comparison
    other_masks = []
    for other_concept in all_concepts:
        if other_concept != concept:
            other_result = model.predict(image_pil, other_concept)
            if len(other_result[0]) > 0:
                other_masks.extend(other_result[0])
    
    best_mask = None
    best_score = float('-inf')
    
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Skip if mask is empty
        if not np.any(mask):
            continue
            
        # Calculate IoU with other concept masks
        current_score = 0
        for other_mask in other_masks:
            if isinstance(other_mask, torch.Tensor):
                other_mask = other_mask.cpu().numpy()
            
            # Calculate IoU
            intersection = np.logical_and(mask, other_mask)
            union = np.logical_or(mask, other_mask)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            
            # Prefer masks with low IoU (less overlap) with other concepts
            current_score -= iou
            
        # Also consider the confidence from logits if available
        if logits is not None and len(logits) > i:
            current_score += logits[i].item()
            
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
    
    return best_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--text_condition', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_idx', type=int, default=0)
    parser.add_argument('--sam_checkpoint', type=str, help='Path to SAM model checkpoint')
    parser.add_argument('--sam_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                      help='SAM model type (default: vit_h)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load and validate input image
    try:
        image_pil = Image.open(args.input_path).convert("RGB")
    except Exception as e:
        print(f"{Fore.RED}Error loading input image: {str(e)}{Fore.RESET}")
        return

    # Initialize SAM model
    try:
        print(f"{Fore.YELLOW}Initializing SAM model (type: {args.sam_type}){Fore.RESET}")
        model = LangSAM(sam_type=args.sam_type, ckpt_path=args.sam_checkpoint)
        print(f"{Fore.GREEN}SAM model initialized successfully{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}Error initializing SAM model: {str(e)}{Fore.RESET}")
        return

    # Split text conditions and process each pair
    text_prompts = args.text_condition.split('+')
    all_concepts = text_prompts
    
    print(f"{Fore.YELLOW}Processing {len(text_prompts)} text prompts{Fore.RESET}")
    
    for text_prompt in text_prompts:
        # Split into individual tokens
        tokens = text_prompt.split()
        if len(tokens) != 2:
            print(f"{Fore.RED}Warning: Expected 2 tokens but got {len(tokens)} in prompt: {text_prompt}{Fore.RESET}")
            continue
            
        first_token, second_token = tokens
        
        try:
            # Get best mask for this concept
            print(f"{Fore.YELLOW}Processing prompt: {text_prompt}{Fore.RESET}")
            mask = get_best_mask_for_concept(model, image_pil, text_prompt, all_concepts)
            
            if mask is not None:
                # Save mask
                mask_path = os.path.join(args.output_path, f"{first_token}_{second_token}.jpg")
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(mask_path)
                print(f"{Fore.GREEN}Saved mask to: {mask_path}{Fore.RESET}")
            else:
                print(f"{Fore.RED}Could not generate valid mask for: {text_prompt}{Fore.RESET}")
                
        except Exception as e:
            print(f"{Fore.RED}Error processing prompt '{text_prompt}': {str(e)}{Fore.RESET}")
            continue

    print(f"{Fore.GREEN}Processing completed{Fore.RESET}")

if __name__ == '__main__':
    main()