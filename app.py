# Load model directly
import torch.utils
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig , pipeline, TrainingArguments
import torch
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import transformers



##Loading model and tokenizer.
compute_dtype = getattr(torch, 'float16')
config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=compute_dtype,bnb_4bit_use_double_quant=False)
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")

# device_map = {"":0}/

model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b",offload_folder='save_folder',quantization_config=config,device_map="auto");
tokenizer.pad_token = tokenizer.eos_token

from functools import partial

def preprocess_batch(batch,tokenizer,max_length=1024):
    
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True
    )
    
def create_prompt_format(sample):
    
    special_command = "You are a chatbot created to answer questions related to ecommerce.Use information in the following paragraph for the same and nothing else.Explain answer only with reference to these paragraphs.If you dont't know, say that you don't know."
    if type(sample) == dict and sample["instruction"] and sample["response"]:
        sys_command = "Below is an instruction that describes a task. Write a response that appropriately completes the request"
        
        prompt = f"\{special_command}n\n### Question: {sample['instruction']}\n ### Answer: {sample['response']}"
        
    else:
        prompt = f"{special_command}"
    return prompt
    
    
    

def create_prompts(data):
    
    prompts = []
    for sample in data:
        prompts.append(create_prompt_format(sample))
    print(prompts[0])
    
    return prompts




dataset = load_dataset("harishvs/ecommerce-faq-llama2-QA",split='train')

# dataset = create_prompts(dataset)

# tokenized_dateset = tokenizer(dataset,return_tensors="pt",padding=True,truncation=True,max_length=1024)

# inputs = tokenized_dateset["input_ids"]
# attention_mask = tokenized_dateset["attention_mask"]

# new_dataset = [{"input_ids": inputs[i], "attention_mask": attention_mask[i]} for i in range(len(tokenized_dateset))]

torch.cuda.empty_cache()
##PEFT config
peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)


##Training arguments
args = TrainingArguments(output_dir='.',evaluation_strategy="no",do_eval=False,per_device_train_batch_size=4,per_gpu_train_batch_size=4,gradient_accumulation_steps=2,save_steps=50,learning_rate=2e-4,group_by_length=True,warmup_ratio=0.03,max_steps=-1,max_grad_norm=0.3,fp16=False,bf16=False)

# model.gradient_checkpointing_enable()

peft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    peft_config=peft_config,
    formatting_func=create_prompts
)

torch.cuda.empty_cache()
##Training model
peft_trainer.train()
new_model = "tuned-dolly"
peft_trainer.model.save_pretrained(new_model)
peft_trainer.tokenizer.save_pretrained(new_model)

