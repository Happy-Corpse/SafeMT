#!/bin/bash
# set up used GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

yaml_files=(
    "./configs/inference/gemma_4b_prompt.yaml"
    "./configs/inference/gemma_4b_dialog.yaml"
)
log_file="log.txt"
> "$log_file"

############# inferecne ###############
for yaml_file in "${yaml_files[@]}"; do
    echo "Processing config: $yaml_file" >> "$log_file"
    python inference/inference.py --config "$yaml_file"  >> "$log_file" 2>&1
done

