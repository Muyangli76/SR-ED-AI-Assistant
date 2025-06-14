# uvicorn app:app --reload
# http://127.0.0.1:8000/docs
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import io
import pdfplumber
import pandas as pd
from docx import Document
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
# Load prompt from external file
with open("prompt_extractor.txt", "r", encoding="utf-8") as f:
    EXTRACTION_PROMPT = f.read()

app = FastAPI()

# Handle text-based input
class TextInput(BaseModel):
    user_input: str

@app.post("/extract-from-text")
async def extract_from_text(input: TextInput):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": input.user_input}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Handle file-based input
def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.file.read())) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file.file.read()))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file.file.read()))
        return df.to_string(index=False)

    else:
        return "⚠️ Unsupported file type. Upload a PDF, DOCX, or CSV."

@app.post("/extract-from-file")
async def extract_from_file(file: UploadFile = File(...)):
    try:
        extracted_text = extract_text_from_file(file)

        if extracted_text.startswith("⚠️"):
            return {"error": extracted_text}

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": extracted_text}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
