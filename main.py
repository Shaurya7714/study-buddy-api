from fastapi import FastAPI
import pdfplumber
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

@app.get("/")
def read_root():
    return {"status": "Study Buddy API is running"}

@app.post("/extract/{note_id}")
def extract_text(note_id: str):
    note = supabase.table("notes").select("*").eq("id", note_id).single().execute().data
    file_bytes = supabase.storage.from_("notes").download(note["file_url"])

    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)

    text = ""
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    supabase.table("notes").update({
        "raw_text": text,
        "status": "extracted"
    }).eq("id", note_id).execute()

    return {"status": "ok", "characters_extracted": len(text)}

import requests

HF_TOKEN = os.environ.get("HF_TOKEN")

@app.post("/summarize/{note_id}")
def summarize(note_id: str):
    note = supabase.table("notes").select("*").eq("id", note_id).single().execute().data
    text = note["raw_text"][:3000]  # keep input manageable

    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )

    result = response.json()

    if isinstance(result, dict) and "error" in result:
        return {"status": "error", "message": result["error"]}

    summary = result[0]["summary_text"]

    supabase.table("summaries").insert({
        "note_id": note_id,
        "summary_text": summary
    }).execute()

    supabase.table("notes").update({"status": "summarized"}).eq("id", note_id).execute()

    return {"status": "ok", "summary": summary}