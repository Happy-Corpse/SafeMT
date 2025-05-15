#!/bin/bash

### file name without .json###
tasks=(
    "2_turn"
)

############# inference ###############
##  API key for inference
api_key=""

# defeine YAML file list for inference
yaml_files=(
   "./configs/api/inference.yaml"
)


############# Inference ###############b
log_file="log.txt"
> "$log_file"

for yaml_file in "${yaml_files[@]}"; do
    for task in "${tasks[@]}"; do
            echo "Processing config: $yaml_file with task: $task" >> "$log_file"
            python data_generation/inference/api_inference.py --config "$yaml_file" --task "$task" --api_key "$api_key" >> "$log_file" 2>&1
        done
done
