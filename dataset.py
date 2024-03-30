import json
from datasets import load_dataset

def create_prompts(data):
    prompts = []
    sys_command = "You are a helpful chatbot trained to answer questions about e-commerce to the user. Be polite and stick to the topic only. Ambiguous or irrelevant questions should be left unanswered.Use the context provided here:"
    for e in data:
        if type(e)==dict and e['context'] and e['instruction'] and e['response']:
            template_prompt = f"[s][INST] [[SYS]]\n{sys_command}{e['context']}\n[[/SYS]]\n\n{e['instruction']} [/INST] {e['response']} [/s]"
            prompts.append(template_prompt)
        else :
            prompts.append(sys_command)
        
    return prompts

dataset = load_dataset("harishvs/ecommerce-faq-llama2-QA",split='train')
prompts = create_prompts(dataset)