#!/bin/bash

# Generated regional sampling script
fused_model="/home/frieren/r13921098/Mix-of-Show/experiments/EDLoRA_cat2_dog6_joint_archived_20241225_194014/models/checkpoint-100/combined_model_joint_concepts"
expdir="custom_regional_sample"

# Set default weights
keypose_condition=''
keypose_adaptor_weight=0.0
sketch_condition="/home/frieren/r13921098/Mix-of-Show/experiments/test/20241226_002123/PromptDataset/spatial_adaptor/2/spatial_mask_2.jpg"
sketch_adaptor_weight=1.0

context_prompt='a photo of <cat1> <cat2> and <dogB1> <dogB2> sleeping'
context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

region1_prompt='[a <cat1> <cat2>, in the scene]'
region1_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
region1='[8, 40, 511, 287]'

region2_prompt='[a <dogB1> <dogB2>, in the scene]'
region2_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
region2='[96, 224, 471, 439]'

prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

python regionally_controlable_sampling.py \
  --pretrained_model=${fused_model} \
  --sketch_adaptor_weight=${sketch_adaptor_weight} \
  --sketch_condition=${sketch_condition} \
  --keypose_adaptor_weight=${keypose_adaptor_weight} \
  --keypose_condition=${keypose_condition} \
  --save_dir="/home/frieren/r13921098/Mix-of-Show/experiments/test/20241226_002123/PromptDataset/validation_edlora_0.7/masks/2" \
  --prompt="${context_prompt}" \
  --negative_prompt="${context_neg_prompt}" \
  --prompt_rewrite="${prompt_rewrite}" \
  --suffix="baseline" \
  --seed=42