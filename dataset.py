import json
from datasets import Dataset
def load_dataset():
    f = open("data.json")
    data = json.load(f)
    return data

def create_prompts(data):
    prompts = []
    for question_instance in data['questions']:
        sys_command = "You are a helpful chatbot trained to answer questions about e-commerce to the user. Be polite and stick to the topic only. Ambiguous or irrelevant questions should be left unanswered."
        template_prompt = f"[s][INST] [[SYS]]\n{sys_command}\n[[/SYS]]\n\n{question_instance['question']} [/INST] {question_instance['answer']} [/s]"
        prompts.append(template_prompt)
    
    dict_prompts = {
        "text": prompts
    }
    return dict_prompts

create_prompts(load_dataset())

