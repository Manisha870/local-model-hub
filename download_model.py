from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch

model_name= "Qwen/Qwen2.5-1.5B-Instruct"
print("Downloading tokenizer...")

tokenizer= AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")

model= AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

model.save_pretrained("./models/qwen2.5-1.5b")
tokenizer.save_pretrained("./models/qwen2.5-1.5b")

print("Model dowloaded and saved!")