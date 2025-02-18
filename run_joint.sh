#!/bin/bash

# Train the joint model
TOTAL_STEPS=800
accelerate launch train_joint_edlora.py -opt options/train/EDLoRA/real/joint_cat2_dog6_config.yml --total_steps $TOTAL_STEPS
# accelerate launch --main_process_port 29501 test_joint_edlora.py -opt options/test/EDLoRA/real/joint_cat2_dog6_test.yml --region_control
# accelerate launch --main_process_port 29501 test_joint_edlora.py -opt options/test/EDLoRA/real/joint_cat2_dog6_test.yml -model_path "$1"
# Sample from the trained model
# If model path is provided as argument, use it; otherwise let the test script find the latest model
