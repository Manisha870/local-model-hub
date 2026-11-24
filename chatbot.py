import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path="./models/qwen2.5-1.5b"

device= "cuda" if torch.cuda.is_available() else "cpu"
print("Loading local model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype= torch.float16 if device == "cuda" else torch.float32,
    local_files_only = True
)

model.to(device)
model.eval()

print("AI is ready!")

messages =[
    {"role": "system", "content": " You are a helpful AI assistant."}
]

while True:
    question = input("\nYou: ")
    if question.lower()== "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })


    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    print("\nAI:", answer)

    messages.append({
        "role": "assistant",
        "content": answer
    })

    if len(messages)>11:
        messages = [messages[0]] + messages[-10:]

print("Goodbye!")