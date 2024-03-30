# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig , pipeline, TrainingArguments
import torch
from dataset import create_prompts,load_dataset
from peft import LoraConfig
from trl import SFTTrainer

##Loading model and tokenizer.
compute_dtype = getattr(torch, 'float16')
config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=compute_dtype,bnb_4bit_use_double_quant=False)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",offload_folder='save_folder');
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

##Switch to GPU
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
    model.to(device)

dataset = load_dataset("harishvs/ecommerce-faq-llama2-QA",split='train')

##PEFT config
peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

##Training arguments
args = TrainingArguments(output_dir='.',evaluation_strategy="no",do_eval=False,per_device_train_batch_size=4,gradient_accumulation_steps=1,save_steps=25,learning_rate=2e-4,group_by_length=True,warmup_ratio=0.03,max_steps=-1,max_grad_norm=0.3,fp16=False,bf16=False)
trainer = SFTTrainer(
    model = model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=args,
    formatting_func=create_prompts
    # dataset_text_field='text',

)

##Training model
trainer.train()
new_model = "tuned-llama-2-7b"
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

