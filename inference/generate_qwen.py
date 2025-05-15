import json
import os
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def build_prompt(processor, item, key, image_path, batch_prompt, processed_batch):
    question = item[key]
    if image_path is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompt = {
            "prompt": prompt,
        }
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        prompt = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        }
        batch_prompt.append(prompt)
        processed_batch.append(item)
        return batch_prompt, processed_batch

def run_qwen_image_features(config, data):
    llm = LLM(
        model=config['model_path'],
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=config['tensor_parallel_size'],
        gpu_memory_utilization=0.9,
        enforce_eager = True
    )
    sampling_params = SamplingParams(
        temperature= 0.0,
        max_tokens=1024,
        skip_special_tokens=True
    )
    processor = AutoProcessor.from_pretrained(config['model_path'])
    total_batches = (len(data) + config['batch_size'] - 1) // config['batch_size']

    batches = tqdm(
        batch_generator(data, config['batch_size']),
        total=total_batches,
        desc="Batch Processing",
        leave=True
    )
    results = []

    for batch in batches:
        batch_prompt = []
        processed_batch = []
        for item in batch:  
            if config['image_dir'] is not None:
                img_path = os.path.join(config['image_dir'], item[config['image_key']].lstrip('/'))
            else:
                img_path = item[config['image_key']]
            if not os.path.exists(img_path):
                continue
            batch_prompt, processed_batch = build_prompt(processor, item, config['query_key'], img_path, batch_prompt, processed_batch)
        outputs = llm.generate(
            prompts=batch_prompt,
            sampling_params=sampling_params,
        )

        for idx, (input_data, output) in enumerate(zip(processed_batch, outputs)):
            result_entry = input_data
            result_entry[config['output_key']] = output.outputs[0].text if output.outputs else ""
            results.append(result_entry)
    return results
