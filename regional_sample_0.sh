#!/bin/bash

# Generated regional sampling script
fused_model="/home/frieren/r13921098/Mix-of-Show/experiments/EDLoRA_cat2_dog6_joint/models/checkpoint-100/combined_model_joint_concepts"
expdir="custom_regional_sample"

# Set default weights
keypose_condition=''
keypose_adaptor_weight=0.0
sketch_condition='/home/frieren/r12921062/Mix-of-Show/datasets/validation_spatial_condition/characters-objects/try5.jpg'
sketch_adaptor_weight=1.0

context_prompt='A <cat1> <cat2> on the right and a <dogB1> <dogB2> on the left'
context_neg_prompt='low quality, extra digit, cropped, worst quality, missing face, missing legs, missing eyes, animals in the background'

region1_prompt='[a <dogB1> <dogB2>, young corgi]'
region1_neg_prompt="[${context_neg_prompt}]"
# region1='[180, 10, 600, 260]'
region1='[0, 90, 510, 485]'

region2_prompt='[a <cat1> <cat2>, gray cat]'
region2_neg_prompt="[${context_neg_prompt}]"
# region2='[240, 480, 720, 720]'
region2='[0, 550, 510, 850]'


prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

python regionally_controlable_sampling.py \
  --pretrained_model=${fused_model} \
  --sketch_adaptor_weight=${sketch_adaptor_weight} \
  --sketch_condition=${sketch_condition} \
  --keypose_adaptor_weight=${keypose_adaptor_weight} \
  --keypose_condition=${keypose_condition} \
  --save_dir="./" \
  --prompt="${context_prompt}" \
  --negative_prompt="${context_neg_prompt}" \
  --prompt_rewrite="${prompt_rewrite}" \
  --suffix="baseline" \
  --seed=42