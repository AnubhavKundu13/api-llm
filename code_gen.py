from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()  # ✅ Make sure this is present

# Load Code-Generating Model
model_path = os.path.abspath("./codegen_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

@app.post("/generate_code/")
async def generate_code(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=150)
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"code": generated_code}
