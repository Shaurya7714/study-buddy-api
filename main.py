# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from supabase import create_client
import os
import requests
import random
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY environment variable is missing.")

supabase = create_client(SUPABASE_URL or "", SUPABASE_SERVICE_KEY or "")

HF_TOKEN = os.environ.get("HF_TOKEN")

@app.get("/")
def read_root():
    return {"status": "Study Buddy API is running"}

@app.post("/extract/{note_id}")
def extract_text(note_id: str):
    try:
        note_res = supabase.table("notes").select("*").eq("id", note_id).single().execute()
        note = note_res.data
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        file_bytes = supabase.storage.from_("notes").download(note["file_url"])

        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        supabase.table("notes").update({
            "raw_text": text,
            "status": "extracted"
        }).eq("id", note_id).execute()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"status": "ok", "characters_extracted": len(text), "text": text}
    except Exception as e:
        supabase.table("notes").update({"status": "error"}).eq("id", note_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

def generate_fallback_summary(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]
    if len(sentences) <= 3:
        return " ".join(sentences) + "."
    return " ".join(sentences[:4]) + "."

@app.post("/summarize/{note_id}")
def summarize(note_id: str):
    try:
        note_res = supabase.table("notes").select("*").eq("id", note_id).single().execute()
        note = note_res.data
        if not note or not note.get("raw_text"):
            raise HTTPException(status_code=400, detail="Note raw text not available for summarization")

        text = note["raw_text"][:3000]

        summary = ""
        if HF_TOKEN:
            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"inputs": text},
                    timeout=15
                )
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    summary = result[0]["summary_text"]
                else:
                    summary = generate_fallback_summary(text)
            except Exception as hf_err:
                print(f"HF Error: {hf_err}")
                summary = generate_fallback_summary(text)
        else:
            summary = generate_fallback_summary(text)

        existing = supabase.table("summaries").select("*").eq("note_id", note_id).execute()
        if existing.data:
            supabase.table("summaries").update({"summary_text": summary}).eq("note_id", note_id).execute()
        else:
            supabase.table("summaries").insert({
                "note_id": note_id,
                "summary_text": summary
            }).execute()

        supabase.table("notes").update({"status": "summarized"}).eq("id", note_id).execute()

        return {"status": "ok", "summary": summary}
    except Exception as e:
        supabase.table("notes").update({"status": "error"}).eq("id", note_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz/{note_id}")
def generate_quiz(note_id: str):
    try:
        summary_res = supabase.table("summaries").select("*").eq("note_id", note_id).execute()
        summary_data = summary_res.data
        
        text_source = ""
        if summary_data and len(summary_data) > 0 and summary_data[0].get("summary_text"):
            text_source = summary_data[0]["summary_text"]
        else:
            note_res = supabase.table("notes").select("*").eq("id", note_id).single().execute()
            text_source = note_res.data.get("raw_text", "")

        if not text_source:
            raise HTTPException(status_code=400, detail="No text available for quiz generation")

        sentences = [s.strip() for s in text_source.replace("\n", " ").split(".") if len(s.strip()) > 20]
        questions = []

        for sentence in sentences[:5]:
            words = [w for w in sentence.split() if len(w) > 4 and w.isalpha()]
            if not words:
                continue
            answer = random.choice(words)
            question_text = sentence.replace(answer, "_______", 1)
            questions.append({
                "note_id": note_id,
                "question": question_text,
                "options": None,
                "correct_answer": answer.strip(".,!?").lower()
            })

        if questions:
            supabase.table("quiz_questions").delete().eq("note_id", note_id).execute()
            supabase.table("quiz_questions").insert(questions).execute()

        supabase.table("notes").update({"status": "ready"}).eq("id", note_id).execute()
        return {"status": "ok", "count": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{note_id}")
def process_note(note_id: str):
    extract_res = extract_text(note_id)
    summarize_res = summarize(note_id)
    quiz_res = generate_quiz(note_id)
    
    return {
        "status": "ready",
        "extracted_chars": extract_res.get("characters_extracted", 0),
        "summary": summarize_res.get("summary", ""),
        "quiz_questions": quiz_res.get("count", 0)
    }