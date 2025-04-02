from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load Quiz-Generating Model
model_path = os.path.abspath("./quizgen_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

@app.post("/generate_quiz/")
async def generate_quiz(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    generated_quiz = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"quiz": generated_quiz}

# Run: uvicorn quizgen_api:app --host 0.0.0.0 --port 8001
