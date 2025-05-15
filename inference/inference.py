import argparse
import os
import json
import torch
import yaml
import ray
from generate_llava import run_llava_image_features
from generate_qwen import run_qwen_image_features
from qwen_chat import run_qwen_chat
from llava_chat import run_llava_chat

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config['input_file']:
        file_path = os.path.join(config['data_path'], f"{config['input_file']}.json")
    else:
        file_path = os.path.join(config['data_path'], f"{args.task}.json")
    with open(file_path) as f:
        queries = json.load(f)
    
    current_dataset_ids = [d['id'] for d in queries]
    model_name = os.path.basename(config['model_path'])
    output_dir = os.path.join(config['output_path'], config['round'])
    output_file = os.path.join(output_dir, f"{model_name}.json")
    try:
        with open(output_file, 'r') as f3:
            results = json.load(f3)
            results = [d for d in results if d['id'] in current_dataset_ids]
    except:
        results = [] 

    processed_key = [d['id'] for d in results]
    queries = [d for d in queries if d['id'] not in processed_key]
    print(f"Process {len(queries)} queries ...")

    ray.init(_temp_dir=None, ignore_reinit_error=True)
    model_type = config['model_path'].split("/")[-1]
    if config['inference_type'] == "single_prompt":
        if model_type[0:4] == "Qwen":
            new_results = run_qwen_image_features(config, queries)
        else:
            new_results = run_llava_image_features(config, queries)
    elif config['inference_type'] == "dialog":
        if model_type[0:4] == "Qwen":
            new_results = run_qwen_chat(config, queries)
        else:
            new_results = run_llava_chat(config, queries)
            
    results.extend(new_results)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nSave path: {output_file}")

    ray.shutdown()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferece Llava")
    parser.add_argument("--config", type=str, default="", help="yaml file path")
    parser.add_argument("--task", type=str, default="", help="file name if needed")
    args = parser.parse_args()

    main(args)