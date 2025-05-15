import argparse
import os
import json
import torch
from tqdm import tqdm
import yaml
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor
import ray
import numpy as np
from prompt import USER_PROMPT
 
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def build_prompt(processor, item, key, image_path, batch_prompt, processed_batch):
    # print(item['id'])
    if image_path is None:
        messages = [
            {
                "role": "user",
                "content": [ {"type": "text", "text": item[key]}]
            }
        ]
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        prompt = {
            "prompt": raw_prompt,
        }
    else:
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "image",
                            "image": image_path,
                        }, {"type": "text", "text": item[key]}]
                    }
                ]
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw_image = Image.open(image_path)
        img_rgb = raw_image.convert('RGB')
        prompt = {
            "prompt": raw_prompt,
            "multi_modal_data": {"image": img_rgb},
        }
    processed_batch.append(item)
    batch_prompt.append(prompt)
    return batch_prompt, processed_batch

def run_llava_image_features(config, queries):
    llm = LLM(
        model=config['model_path'],
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=config['tensor_parallel_size'],
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager = True,
        max_model_len=4096, ## for llama not for llava
        max_num_seqs=20   ## for llama not for llava
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        skip_special_tokens=True
    )
    processor = AutoProcessor.from_pretrained(config['model_path'],trust_remote_code=True)
    total_batches = (len(queries) + config['batch_size'] - 1) // config['batch_size']

    batches = tqdm(
        batch_generator(queries, config['batch_size']),
        total=total_batches,
        desc="Batch Processing",
        leave=True
    )
    results = []

    for batch in batches:
        batch_prompt = []
        processed_batch = []
        for item in batch:  
            if config['image_dir']:
                img_path = os.path.join(config['image_dir'], item[config['image_key']].lstrip('/'))
            else:
                img_path = item[config['image_key']]
            if not os.path.exists(img_path):
                # raise FileNotFoundError(f"Missing Image: {img_path}")
                continue
            if config['query_key'] in item.keys():
                batch_prompt, processed_batch = build_prompt(processor, item, config['query_key'], img_path, batch_prompt, processed_batch)
      

        outputs = llm.generate(
            prompts=batch_prompt,
            sampling_params=sampling_params,
        )

        for idx, (input_data, output) in enumerate(zip(processed_batch, outputs)):
            input_data[config['output_key']] = output.outputs[0].text if output.outputs else ""
            results.append(input_data)

    return results
