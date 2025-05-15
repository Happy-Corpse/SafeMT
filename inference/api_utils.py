from tqdm import tqdm
import os
import base64
import copy

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def message_generation(text, key, image_url):
    if key == "question":
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": text[key]
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url  # 使用base64或网络URL
                            }
                        }
                    ]
                }
            ]
    elif key == "dialogue":
        dialog = copy.deepcopy(text[key])
        for idx, t in enumerate(dialog):
            if idx == 0:
                t['content'] = [
                        {
                            "type": "image_url",
                            "image_url": {"url":image_url},
                        },
                        {"type": "text", "text": t['content']},
                    ]
            else:
                t['content'] = [
                        {"type": "text", "text": t['content']},
                    ]
        messages = dialog
    return messages


def generation_prompt(batch, config):
    batch_prompt = []
    for text in tqdm(batch, desc=f"Processing", total=len(batch)):
        image_url = None
        if config['image_dir'] is not None:
            image_path = os.path.join(config['image_dir'],text[config['image_key']].lstrip('/'))
        else:
            image_path = text[config['image_key']]
        if image_path and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            image_url = f"data:image/jpeg;base64,{base64_image}"
        else:
            print(image_path)
        messages = message_generation(text, config['query_key'], image_url)
        batch_prompt.append(messages)
    return batch_prompt