import transformers
import torch

print(torch.cuda.is_available())
print(transformers.is_bitsandbytes_available())