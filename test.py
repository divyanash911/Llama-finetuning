# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="tuned-llama-2-7b")

# Let's generate some text
prompt = "How can I create an account?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


