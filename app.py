# http://127.0.0.1:8000/docs
# uvicorn app:app --reload

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the prompt from file
def load_prompt(filepath="prompt_extractor.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

EXTRACTION_PROMPT = load_prompt()

app = FastAPI()

class TextInput(BaseModel):
    user_input: str

@app.post("/extract-from-text")
async def extract_from_text(input: TextInput):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": input.user_input}
            ],
            temperature=0.4
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/extract-from-file")
async def extract_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8", errors="ignore")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.4
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
