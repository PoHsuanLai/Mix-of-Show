import argparse
import hashlib
import json
import os.path

import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import T2IAdapter
from PIL import Image

from mixofshow.pipelines.pipeline_regionally_t2iadapter import RegionallyT2IAdapterPipeline


def sample_image(pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    sketch_adaptor_weight=1.0,
    region_sketch_adaptor_weight='',
    keypose_adaptor_weight=1.0,
    region_keypose_adaptor_weight='',
    **extra_kargs
):

    keypose_condition = extra_kargs.pop('keypose_condition')
    if keypose_condition is not None:
        keypose_adapter_input = [keypose_condition] * len(input_prompt)
    else:
        keypose_adapter_input = None

    sketch_condition = extra_kargs.pop('sketch_condition')
    if sketch_condition is not None:
        sketch_adapter_input = [sketch_condition] * len(input_prompt)
    else:
        sketch_adapter_input = None

    images = pipe(
        prompt=input_prompt,
        negative_prompt=input_neg_prompt,
        keypose_adapter_input=keypose_adapter_input,
        keypose_adaptor_weight=keypose_adaptor_weight,
        region_keypose_adaptor_weight=region_keypose_adaptor_weight,
        sketch_adapter_input=sketch_adapter_input,
        sketch_adaptor_weight=sketch_adaptor_weight,
        region_sketch_adaptor_weight=region_sketch_adaptor_weight,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        **extra_kargs).images
    return images


def build_model(pretrained_model, device):
    pipe = RegionallyT2IAdapterPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device)
    assert os.path.exists(os.path.join(pretrained_model, 'new_concept_cfg.json'))
    with open(os.path.join(pretrained_model, 'new_concept_cfg.json'), 'r') as json_file:
        new_concept_cfg = json.load(json_file)
    pipe.set_new_concept_cfg(new_concept_cfg)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model, subfolder='scheduler')
    pipe.keypose_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_openpose_sd14v1', torch_dtype=torch.float16).to(device)
    pipe.sketch_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_sketch_sd14v1', torch_dtype=torch.float16).to(device)
    return pipe


def prepare_text(prompt, region_prompts, height, width):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    region_collection = []

    regions = region_prompts.split('|')
    print(f"\nProcessing regions with height={height}, width={width}")

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        pos = eval(pos)
        print(f"\nRegion prompt: {prompt_region}")
        print(f"Original position: {pos}")
        
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
            print(f"Using default position: {pos}")
        else:
            # The coordinates are already in pixel space (0 to height/width)
            # We need to normalize them to 0-1 range
            pos[0], pos[2] = pos[0] / height, pos[2] / height  # y coordinates
            pos[1], pos[3] = pos[1] / width, pos[3] / width    # x coordinates
            print(f"Normalized position: {pos}")

        region_collection.append((prompt_region, neg_prompt_region, pos))
    return (prompt, region_collection)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='experiments/composed_edlora/anythingv4/hina+kario+tezuka+mitsuha+son_anythingv4/combined_model_base', type=str)
    parser.add_argument('--sketch_condition', default=None, type=str)
    parser.add_argument('--sketch_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_sketch_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--prompt', default='photo of a toy', type=str)
    parser.add_argument('--negative_prompt', default='', type=str)
    parser.add_argument('--prompt_rewrite', default='', type=str)
    parser.add_argument('--seed', default=16141, type=int)
    parser.add_argument('--suffix', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = build_model(args.pretrained_model, device)

    if args.sketch_condition is not None and os.path.exists(args.sketch_condition):
        sketch_condition = Image.open(args.sketch_condition).convert('L')
        # Get original dimensions
        orig_width, orig_height = sketch_condition.size
        print(f'Original sketch size: {orig_width}x{orig_height}')
        
        # If the sketch is doubled width (1024x512), resize it to 512x512 for the model
        if orig_width == 1024 and orig_height == 512:
            print('Detected doubled width sketch, resizing to 512x512 for model input')
            sketch_condition = sketch_condition.resize((512, 512), Image.Resampling.LANCZOS)
            # Set the target width to 1024 to maintain doubled width in output
            width_sketch, height_sketch = 1024, 512
        else:
            # For other cases, resize to 512x512
            if sketch_condition.size != (512, 512):
                print(f'Resizing sketch from {sketch_condition.size} to 512x512')
                sketch_condition = sketch_condition.resize((512, 512), Image.Resampling.LANCZOS)
            width_sketch, height_sketch = 512, 512
        
        print('Using sketch condition')
        print(f'Target output dimensions: {width_sketch}x{height_sketch}')
    else:
        sketch_condition, width_sketch, height_sketch = None, 0, 0
        print('skip sketch condition')

    if args.keypose_condition is not None and os.path.exists(args.keypose_condition):
        keypose_condition = Image.open(args.keypose_condition).convert('RGB')
        if keypose_condition.size != (width_sketch, height_sketch) and width_sketch != 0:
            print(f'Resizing keypose from {keypose_condition.size} to {width_sketch}x{height_sketch}')
            keypose_condition = keypose_condition.resize((width_sketch, height_sketch), Image.Resampling.LANCZOS)
        width_pose, height_pose = keypose_condition.size
        print('use pose condition')
    else:
        keypose_condition, width_pose, height_pose = None, 0, 0
        print('skip pose condition')

    if width_sketch != 0 and width_pose != 0:
        assert width_sketch == width_pose and height_sketch == height_pose, 'conditions should be same size'
    width, height = max(width_pose, width_sketch), max(height_pose, height_sketch)
    print(f'Using dimensions for region scaling: {width}x{height}')
    
    kwargs = {
        'sketch_condition': sketch_condition,
        'keypose_condition': keypose_condition,
        'height': height,
        'width': width,
    }

    prompts = [args.prompt]
    prompts_rewrite = [args.prompt_rewrite]
    input_prompt = [prepare_text(p, p_w, height, width) for p, p_w in zip(prompts, prompts_rewrite)]
    save_prompt = input_prompt[0][0]

    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        sketch_adaptor_weight=args.sketch_adaptor_weight,
        region_sketch_adaptor_weight=args.region_sketch_adaptor_weight,
        keypose_adaptor_weight=args.keypose_adaptor_weight,
        region_keypose_adaptor_weight=args.region_keypose_adaptor_weight,
        **kwargs)

    print(f'save to: {args.save_dir}')

    configs = [
        f'pretrained_model: {args.pretrained_model}\n',
        f'context_prompt: {args.prompt}\n', f'neg_context_prompt: {args.negative_prompt}\n',
        f'sketch_condition: {args.sketch_condition}\n', f'sketch_adaptor_weight: {args.sketch_adaptor_weight}\n',
        f'region_sketch_adaptor_weight: {args.region_sketch_adaptor_weight}\n',
        f'keypose_condition: {args.keypose_condition}\n', f'keypose_adaptor_weight: {args.keypose_adaptor_weight}\n',
        f'region_keypose_adaptor_weight: {args.region_keypose_adaptor_weight}\n', f'random seed: {args.seed}\n',
        f'prompt_rewrite: {args.prompt_rewrite}\n'
    ]
    hash_code = hashlib.sha256(''.join(configs).encode('utf-8')).hexdigest()[:8]

    save_prompt = save_prompt.replace(' ', '_')
    save_name = f'{save_prompt}---{args.suffix}---{hash_code}.png'
    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    save_path = os.path.join(save_dir, save_name)
    save_config_path = os.path.join(save_dir, save_name.replace('.png', '.txt'))

    os.makedirs(save_dir, exist_ok=True)
    image[0].save(os.path.join(save_dir, save_name))

    with open(save_config_path, 'w') as fw:
        fw.writelines(configs)