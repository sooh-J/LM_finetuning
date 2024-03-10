#!bin/bash
A="0,1,2,3,4,5,6,7"
C=8 # length of A

CUDA_VISIBLE_DEVICES=$A
TENSOR_PARALLEL_SIZE=$C

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opt_peft/src/finetune_opt_bnb_peft.py \
    --output_dir  opt_peft/results \
    --train_path /home/taeyoon/nas2/Medical-intern/data/ADNI/train_data.json \
    --eval_path /home/taeyoon/nas2/Medical-intern/data/ADNI/val_data.json \
    --input ADNI_patient_description # [patient_description, w_rationales] 
