#---------------------------------------------anime-------------------------------------------

real_character=1
seed=42

# fused_model='/home/frieren/r12921062/Mix-of-Show/experiments/composed_edlora/chilloutmix/cat+dog/combined_model_base'
fused_model="/home/frieren/r13921098/Mix-of-Show/experiments/EDLoRA_cat2_dog6_joint/models/checkpoint-100/combined_model_joint_concepts/"
expdir="org"
keypose_condition=''
# keypose_condition='experiments/composed_edlora/chilloutmix/potter+hermione+thanos_chilloutmix/combined_model_base'
keypose_adaptor_weight=1.0

# sketch_condition='/home/frieren/r12921062/Mix-of-Show/datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
sketch_condition='/home/frieren/r12921062/Mix-of-Show/datasets/validation_spatial_condition/characters-objects/try5.jpg'
# sketch_condition='/home/frieren/r13921098/Mix-of-Show/experiments/test/20241226_002851/PromptDataset/spatial_adaptor/0/spatial_mask_0.jpg'
# sketch_condition="/home/frieren/r13921098/Mix-of-Show/experiments/test/20241225_184042/PromptDataset/spatial_adaptor/0/spatial_mask_0.jpg"
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
# context_prompt='A <cat1> <cat2> on the right and a <dogB1> <dogB2> on the left'
# context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

# region1_prompt='[a <cat1> <cat2>, in the scene]'
# region1_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
# region1='[24, 48, 503, 510]'

# region2_prompt='[a <dogB1> <dogB2>, in the scene]'
# region2_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
# region2='[16, 576, 495, 958]'

prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

python regionally_controlable_sampling.py \
  --pretrained_model=${fused_model} \
  --sketch_adaptor_weight=${sketch_adaptor_weight}\
  --sketch_condition=${sketch_condition} \
  --keypose_adaptor_weight=${keypose_adaptor_weight}\
  --keypose_condition=${keypose_condition} \
  --save_dir="./billy/up" \
  --prompt="${context_prompt}" \
  --negative_prompt="${context_neg_prompt}" \
  --prompt_rewrite="${prompt_rewrite}" \
  --suffix="baseline" \
  --seed=$seed

# Generated regional sampling script
# fused_model="/home/frieren/r13921098/Mix-of-Show/experiments/EDLoRA_cat2_dog6_joint_archived_20241225_194014/models/checkpoint-100/combined_model_joint_concepts"
fused_model='/home/frieren/r12921062/Mix-of-Show/experiments/composed_edlora/chilloutmix/cat+dog/combined_model_base'
expdir="custom_regional_sample"

# Set default weights
keypose_condition=''
keypose_adaptor_weight=0.0
# sketch_condition='/home/frieren/r13921098/Mix-of-Show/experiments/test/20241226_002851/PromptDataset/spatial_adaptor/0/spatial_mask_0.jpg'
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

# context_prompt='A <cat1> <cat2> on the right and a <dogB1> <dogB2> on the left'
# context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

# region1_prompt='[a <cat1> <cat2>, in the scene]'
# region1_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
# region1='[24, 48, 503, 510]'

# region2_prompt='[a <dogB1> <dogB2>, in the scene]'
# region2_neg_prompt='[longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality]'
# region2='[16, 576, 495, 958]'

prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

python regionally_controlable_sampling.py \
  --pretrained_model=${fused_model} \
  --sketch_adaptor_weight=${sketch_adaptor_weight} \
  --sketch_condition=${sketch_condition} \
  --keypose_adaptor_weight=${keypose_adaptor_weight} \
  --keypose_condition=${keypose_condition} \
  --save_dir="./billy/down" \
  --prompt="${context_prompt}" \
  --negative_prompt="${context_neg_prompt}" \
  --prompt_rewrite="${prompt_rewrite}" \
  --suffix="baseline" \
  --seed=42