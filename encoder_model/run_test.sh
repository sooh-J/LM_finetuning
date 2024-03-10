#!bin/bash
A="0,1,2,3,4,5,6,7"
C=8 # length of A

CUDA_VISIBLE_DEVICES=$A
TENSOR_PARALLEL_SIZE=$C


# for P (data from GPT)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python encoder_model/test_clinical-longformer.py \
    --output_dir encoder_model/results/Clinical-Longformer-final \
    --train_path data/train_rationale/train_parsed.json \
    --eval_path data/train_rationale/valid_parsed.json \
    --input w_rationales # [patient_description, w_rationales] 

# # for P (oversampling from student model)
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python /home/taeyoon/nas2/Medical-intern/Clinical-Longformer-final/train_clinical-longformer.py \
#     --output_dir /home/taeyoon/nas2/Medical-intern/Clinical-Longformer-final \
#     --train_path /home/taeyoon/nas2/Medical-intern/data/train_rationale/train_parsed.json \
#     --eval_path /home/taeyoon/nas2/Medical-intern/data/train_rationale/valid_parsed.json \
#     --input student_patient_description # [patient_description, w_rationales] 


# # for P+R
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python /home/taeyoon/nas2/Medical-intern/Clinical-Longformer-final/train_clinical-longformer.py \
#     --output_dir /home/taeyoon/nas2/Medical-intern/Clinical-Longformer-final \
#     --train_path /home/taeyoon/nas2/Medical-intern/data/train_rationale/train_parsed.json \
#     --eval_path /home/taeyoon/nas2/Medical-intern/data/train_rationale/valid_parsed.json \
#     --input w_rationales # [patient_description, w_rationales] 