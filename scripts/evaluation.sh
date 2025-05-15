#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6,7

### file name without .json###
tasks=(
    "gemma-3-4b-it"
)

############# inference ###############
##  define Python file path

api_key=""

# defeine YAML file list for inference
yaml_files=(
   "./configs/api/evaluation.yaml"
)


############# evaluation ###############
log_file="log.txt"
> "$log_file"

for yaml_file in "${yaml_files[@]}"; do
    for task in "${tasks[@]}"; do
            echo "Processing config: $yaml_file with task: $task" >> "$log_file"
            python evaluation/api_evaluation.py --config "$yaml_file" --task "$task" --api_key "$api_key" >> "$log_file" 2>&1
        done
done
