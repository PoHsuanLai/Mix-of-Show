# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/1111_EDLoRA_cat_Cmix_B4_Repeat500.yml
accelerate launch --main_process_port 10000 train_edlora.py -opt options/train/EDLoRA/real/1112_EDLoRA_dog_Cmix_B4_Repeat500.yml
accelerate launch --main_process_port 10001 train_edlora.py -opt options/train/EDLoRA/real/1113_EDLoRA_dog6_Cmix_B4_Repeat500.yml
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/1114_EDLoRA_vase_Cmix_B4_Repeat500.yml
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/1115_EDLoRA_pet_cat1_Cmix_B4_Repeat500.yml
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/1116_EDLoRA_flower_Cmix_B4_Repeat500.yml
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/1117_EDLoRA_glasses_Cmix_B4_Repeat500.yml