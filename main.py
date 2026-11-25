from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
)


@app.post("/text", response_class=PlainTextResponse)
async def receive_text(request: Request) -> str:
    body = (await request.body()).decode("utf-8")
    messages = [
            {
                "role": "system",
                "content": "Reply concisely and helpfully to the user's input.",
            },
            {"role": "user", "content": body},
        ]
    output = generator(
            messages, max_new_tokens=10000, do_sample=True, temperature=0.7
        )
    return output[0]["generated_text"][-1]["content"]