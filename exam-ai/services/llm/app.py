from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)

app = FastAPI()

class Payload(BaseModel):
    student_id: str
    seat_id: str
    window_sec: int
    events: list

@app.post("/summarize")
def summarize(p: Payload):
    prompt = f"""
You are an exam invigilation assistant.
ONLY use facts from the JSON. Do NOT invent events.
If evidence is weak, output "Needs Review".

Return in this format:
risk_class: ...
reasons:
- ...
evidence_to_review_first: ...

JSON:
{json.dumps(p.model_dump(), indent=2)}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"response": text}
