import copy


def load_new_concept(pipe, new_concept_embedding, enable_edlora=True):
    new_concept_cfg = {}

    for idx, (concept_name, concept_embedding) in enumerate(new_concept_embedding.items()):
        if enable_edlora:
            num_new_embedding = 16
        else:
            num_new_embedding = 1
        new_token_names = [f'<new{idx * num_new_embedding + layer_id}>' for layer_id in range(num_new_embedding)]
        num_added_tokens = pipe.tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == len(new_token_names), 'some token is already in tokenizer'
        new_token_ids = [pipe.tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        # init embedding
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        token_embeds[new_token_ids] = concept_embedding.clone().to(token_embeds.device, dtype=token_embeds.dtype)
        print(f'load embedding: {concept_name}')

        new_concept_cfg.update({
            concept_name: {
                'concept_token_ids': new_token_ids,
                'concept_token_names': new_token_names
            }
        })

    return pipe, new_concept_cfg


def merge_lora_into_weight(original_state_dict, lora_state_dict, model_type, alpha):
    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj.lora_down.weight') \
                .replace('k_proj.weight', 'k_proj.lora_down.weight') \
                .replace('v_proj.weight', 'v_proj.lora_down.weight') \
                .replace('out_proj.weight', 'out_proj.lora_down.weight') \
                .replace('fc1.weight', 'fc1.lora_down.weight') \
                .replace('fc2.weight', 'fc2.lora_down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q.lora_down.weight') \
                .replace('to_k.weight', 'to_k.lora_down.weight') \
                .replace('to_v.weight', 'to_v.lora_down.weight') \
                .replace('to_out.0.weight', 'to_out.0.lora_down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.lora_down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2.lora_down.weight') \
                .replace('proj_out.weight', 'proj_out.lora_down.weight') \
                .replace('proj_in.weight', 'proj_in.lora_down.weight')

        return lora_down_name

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)

    load_cnt = 0
    for k in new_state_dict.keys():
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            lora_down_params = lora_state_dict[lora_down_name].to(original_params.device)
            lora_up_params = lora_state_dict[lora_up_name].to(original_params.device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze() @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    print(f'load {load_cnt} LoRAs of {model_type}')
    return new_state_dict


def convert_edlora(pipe, state_dict, enable_edlora, alpha=0.6):
    # Extensive debugging for state_dict
    print("Debug: convert_edlora input state_dict type:", type(state_dict))
    
    # If state_dict is a tensor or has a keys method, print its keys
    if hasattr(state_dict, 'keys'):
        print("Debug: state_dict keys:", list(state_dict.keys()))
    
    # Handle cases where state_dict might be None or not have expected keys
    if state_dict is None:
        print("Warning: state_dict is None. Returning original pipeline.")
        return pipe, {}

    # Try to handle different possible input formats
    try:
        # First, try to access 'params' key if it exists
        if isinstance(state_dict, dict) and 'params' in state_dict:
            state_dict = state_dict['params']
        
        # If state_dict is still None or empty, return
        if not state_dict:
            print("Warning: state_dict is empty after extraction.")
            return pipe, {}

        # Print keys after potential extraction
        print("Debug: Extracted state_dict keys:", list(state_dict.keys()))
    except Exception as e:
        print(f"Error extracting state_dict: {e}")
        return pipe, {}

    # Initialize new_concept_cfg to an empty dictionary
    new_concept_cfg = {}

    # step 1: load embedding
    # Add more detailed error handling and logging
    try:
        if 'new_concept_embedding' in state_dict and state_dict['new_concept_embedding']:
            print("Debug: Found new_concept_embedding")
            pipe, new_concept_cfg = load_new_concept(pipe, state_dict['new_concept_embedding'], enable_edlora)
        else:
            print("Warning: No new concept embedding found in state_dict.")
            print("Debug: Available keys:", list(state_dict.keys()))
    except Exception as e:
        print(f"Error loading new concept embedding: {e}")

    # step 2: merge lora weight to unet
    try:
        if 'unet' in state_dict:
            unet_lora_state_dict = state_dict['unet']
            pretrained_unet_state_dict = pipe.unet.state_dict()
            updated_unet_state_dict = merge_lora_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet', alpha=alpha)
            pipe.unet.load_state_dict(updated_unet_state_dict)
        else:
            print("Warning: No unet LoRA weights found in state_dict.")
            print("Debug: Available keys:", list(state_dict.keys()))
    except Exception as e:
        print(f"Error merging unet LoRA weights: {e}")

    # step 3: merge lora weight to text_encoder
    try:
        if 'text_encoder' in state_dict:
            text_encoder_lora_state_dict = state_dict['text_encoder']
            pretrained_text_encoder_state_dict = pipe.text_encoder.state_dict()
            updated_text_encoder_state_dict = merge_lora_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', alpha=alpha)
            pipe.text_encoder.load_state_dict(updated_text_encoder_state_dict)
        else:
            print("Warning: No text encoder LoRA weights found in state_dict.")
            print("Debug: Available keys:", list(state_dict.keys()))
    except Exception as e:
        print(f"Error merging text encoder LoRA weights: {e}")

    return pipe, new_concept_cfg
