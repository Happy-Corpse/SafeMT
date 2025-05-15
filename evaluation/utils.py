from prompt import *
from tqdm.asyncio import tqdm

def evaluation_prompt(data, key, task):
    batch_messages = []
    for text in tqdm(data, desc=f"Processing {task}", total=len(data)):
        try:
            if text['legal_type'] == 'Political':
                gpt_query_template = gpt_query_template_safety_political_lobbing
                print('Political')
            elif text['legal_type'] == 'Legal Opinion':
                gpt_query_template = gpt_query_template_safety_legal
                print('Legal Opinion')
            elif text['legal_type'] == 'Financial Advice':
                gpt_query_template = gpt_query_template_safety_financial
                print('Financial Advice')
            elif text['legal_type'] == 'Health Consultation':
                gpt_query_template = gpt_query_template_safety_health
                print('Health Consultation')
            else:
                gpt_query_template = gpt_query_template_safety
        except:
            gpt_query_template = gpt_query_template_safety
        if key == "Dialog_Response":
            dialogue = text['dialogue']
        else:
            dialogue = ""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": gpt_query_template.format(question = text['question'], dialog=dialogue, prediction=text[key], key=key)
                    }

                ]
            }
        ]
        batch_messages.append(messages)
    return batch_messages




