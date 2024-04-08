import json
from datasets import load_dataset
from langchain import PromptTemplate


dataset = load_dataset("harishvs/ecommerce-faq-llama2-QA",split='train')
prompts = create_prompts(dataset)