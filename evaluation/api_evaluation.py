import json
import os
import httpx
import asyncio
from tqdm.asyncio import tqdm
import yaml
import argparse
from datetime import datetime
from utils import *
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None)
parser.add_argument('--api_key', type=str, default=None)
parser.add_argument('--task', type=str, default=None)

args = parser.parse_args()

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

input_dir = config['input_dir']
input_file = os.path.join(input_dir, f"{args.task}.json")

output_dir = config['output_dir']
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"{args.task}.json")

log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_path = os.path.join(output_dir, f'log_{log_timestamp}.txt')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
model_name = config['model_id']
print(model_name)

headers = {
    'Authorization': args.api_key,
    'Content-Type': 'application/json',
}

CONCURRENCY = config.get('concurrency')
semaphore = asyncio.Semaphore(CONCURRENCY)

async def get_batch_response_with_retry(messages, n, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await get_batch_response(messages, n)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))

async def get_batch_response(messages, n):
    all_translated = []
    async with semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0)) as client:
            tasks = []
            for message in messages:
                # print(message)
                json_data = {
                    'model': model_name,
                    'messages': message,
                    'temperature': config.get('temperature', 1.0),
                    'stream': False,
                }
                task = asyncio.create_task(send_request(client, json_data, n))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            for result in results:
                all_translated.append(result)
    return all_translated

async def send_request(client, json_data, n):
    prompt_answers = []
    for _ in range(n):
        response = await client.post(config['base_url'], headers=headers, json=json_data)
        try:
            result = response.json()
            try:
                ans = result['choices'][0]['message']['content']
            except: 
                ans = result
            prompt_answers.append(ans)
        except Exception as e:
            error_msg = str(e)
            prompt_answers.append(error_msg)
            
            # 记录错误信息到日志
            with open(log_path, 'a') as log_file:
                log_file.write(f"Request Error: {error_msg}\n")
                log_file.write("-" * 40 + "\n")
    return prompt_answers

def save_json(data, file_name):
    json_data = json.dumps(data, indent=2, ensure_ascii=False)
    with open(file_name, 'w') as file:
        file.write(json_data)
    file.close()

def split_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

async def process_tasks(input_path):
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    current_dataset_ids = [d[config["unique_key"]] for d in dataset]

    try:
        with open(output_file, 'r') as f3:
            results = json.load(f3)
            # results = [d for d in results if d['topic'] != '' and d['question'] != '' and d['answer'] != '']
            results = [d for d in results if d['id'] in current_dataset_ids]
    except:
        results = []
    
    processed_key = [d[config["unique_key"]] for d in results]

    
    dataset = [d for d in dataset if d[config["unique_key"]] not in processed_key]
    
    print(f'to be processed: {len(dataset)}')
    
    tasks = []
    total_batches = (len(dataset) + config['batch_size'] - 1) // config['batch_size']
    pbar = tqdm(total=total_batches, desc="Processing batches")
    
    batches = tqdm(
        split_into_batches(dataset, config['batch_size']),
        total=total_batches,
        desc="Batch Processing",
        leave=True
    )

    for batch in batches:
        messages = evaluation_prompt(batch, config['query_key'], args.task)

        task = asyncio.create_task(get_batch_response_with_retry(messages, n=config['n']))
        tasks.append(task)

    
    response_message_batches = await asyncio.gather(*tasks)
    for i, response_message_batch in enumerate(response_message_batches):
        pbar.update(1)
        start_index = i * config['batch_size']
        for j, converted_item in enumerate(response_message_batch):
            dataset_index = start_index + j
            if dataset_index < len(dataset):
                response = converted_item[0]
                dataset[dataset_index][config['save_key']] = response if response else ""
        
        results.extend(dataset[start_index:start_index + len(response_message_batch)])
        save_json(results, output_file)
    
    pbar.close()


async def main():
    # 初始化日志文件
    with open(log_path, 'w') as log_file:
        log_file.write(f"Date: {asyncio.get_event_loop().time()}\n")
        log_file.write("=" * 50 + "\n\n")
    
    await process_tasks(input_file)
    
if __name__ == "__main__":
    asyncio.run(main())
