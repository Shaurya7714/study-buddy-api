# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from supabase import create_client
import os
import requests
import random
import math
from datetime import datetime, timedelta, timezone
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

# ─── Healthcheck ────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "Study Buddy API is running"}

# ─── Text Extraction ────────────────────────────────────────────

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

# ─── Summarization ──────────────────────────────────────────────

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
                    timeout=30
                )
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    summary = result[0]["summary_text"]
                else:
                    summary = generate_fallback_summary(text)
            except Exception as hf_err:
                print(f"HF Summarization Error: {hf_err}")
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

# ─── AI Question Generation (T5 model) ─────────────────────────

def generate_ai_questions(text: str):
    """Try to generate real questions using T5 QG model via Hugging Face."""
    if not HF_TOKEN:
        return None

    try:
        # Format input for the T5 QG model: highlight a sentence, ask it to generate a question
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
        ai_questions = []

        for sentence in sentences[:5]:
            # Find key words to use as answer targets
            words = [w for w in sentence.split() if len(w) > 4 and w.isalpha()]
            if not words:
                continue

            answer = random.choice(words)
            # T5 QG input format: "answer: <answer> context: <context>"
            prompt = f"answer: {answer} context: {sentence}"

            response = requests.post(
                "https://api-inference.huggingface.co/models/valhalla/t5-base-qg-hl",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt},
                timeout=20
            )
            result = response.json()

            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                question_text = result[0]["generated_text"].strip()
                if question_text and len(question_text) > 10:
                    ai_questions.append({
                        "question": question_text,
                        "correct_answer": answer.strip(".,!?").lower()
                    })

        return ai_questions if len(ai_questions) > 0 else None
    except Exception as e:
        print(f"AI QG Error: {e}")
        return None

# ─── Multiple-Choice Quiz Generation ───────────────────────────

def pick_distractors(correct_answer: str, all_words: list, count: int = 3) -> list:
    """Pick distractor words that are similar length to the correct answer."""
    correct_len = len(correct_answer)
    candidates = [
        w for w in all_words
        if w.lower() != correct_answer.lower()
        and w.isalpha()
        and len(w) > 3
        and abs(len(w) - correct_len) <= 3
    ]
    # Deduplicate
    candidates = list(set(candidates))
    random.shuffle(candidates)
    distractors = candidates[:count]

    # If not enough distractors, pad with generic words
    fallback_words = ["example", "process", "system", "function", "method", "concept", "element",
                      "structure", "analysis", "theory", "model", "factor", "result", "approach"]
    while len(distractors) < count:
        fallback = random.choice(fallback_words)
        if fallback.lower() != correct_answer.lower() and fallback not in distractors:
            distractors.append(fallback)

    return [d.lower() for d in distractors]


@app.post("/generate-quiz/{note_id}")
def generate_quiz(note_id: str):
    try:
        # Get text source (summary preferred, raw text fallback)
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

        # Collect all meaningful words for distractor pool
        all_words = [w for w in text_source.replace("\n", " ").split() if len(w) > 3 and w.isalpha()]

        # Try AI question generation first, fall back to cloze
        ai_qs = generate_ai_questions(text_source)
        questions = []

        if ai_qs:
            # AI-generated questions
            for aq in ai_qs:
                distractors = pick_distractors(aq["correct_answer"], all_words)
                options = distractors + [aq["correct_answer"].lower()]
                random.shuffle(options)
                questions.append({
                    "note_id": note_id,
                    "question": aq["question"],
                    "options": options,
                    "correct_answer": aq["correct_answer"].lower()
                })
        else:
            # Fallback: cloze-style with multiple choice
            sentences = [s.strip() for s in text_source.replace("\n", " ").split(".") if len(s.strip()) > 20]

            for sentence in sentences[:5]:
                words = [w for w in sentence.split() if len(w) > 4 and w.isalpha()]
                if not words:
                    continue
                answer = random.choice(words)
                question_text = sentence.replace(answer, "_______", 1)

                distractors = pick_distractors(answer, all_words)
                options = distractors + [answer.strip(".,!?").lower()]
                random.shuffle(options)

                questions.append({
                    "note_id": note_id,
                    "question": question_text,
                    "options": options,
                    "correct_answer": answer.strip(".,!?").lower()
                })

        if questions:
            supabase.table("quiz_questions").delete().eq("note_id", note_id).execute()
            supabase.table("quiz_questions").insert(questions).execute()

        supabase.table("notes").update({"status": "ready"}).eq("id", note_id).execute()
        return {"status": "ok", "count": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Process Pipeline ───────────────────────────────────────────

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

# ─── SM-2 Spaced Repetition ────────────────────────────────────

def sm2_algorithm(quality: int, easiness_factor: float, interval: int, repetitions: int):
    """
    SM-2 algorithm implementation.
    quality: 0-5 (0=complete fail, 5=perfect recall)
    Returns: (new_ef, new_interval, new_repetitions)
    """
    # Update easiness factor
    new_ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)  # EF must never go below 1.3

    if quality >= 3:
        # Correct response
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = math.ceil(interval * new_ef)
        new_repetitions = repetitions + 1
    else:
        # Incorrect — reset
        new_interval = 1
        new_repetitions = 0

    return new_ef, new_interval, new_repetitions


@app.post("/review/{user_id}/{question_id}")
def review_question(user_id: str, question_id: str, quality: int = 5):
    """Update spaced repetition progress after a review."""
    try:
        if quality < 0 or quality > 5:
            raise HTTPException(status_code=400, detail="Quality must be between 0 and 5")

        # Get existing progress or defaults
        existing = supabase.table("user_question_progress") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("question_id", question_id) \
            .execute()

        if existing.data and len(existing.data) > 0:
            progress = existing.data[0]
            ef = progress["easiness_factor"]
            interval = progress["interval_days"]
            reps = progress["repetitions"]
        else:
            ef = 2.5
            interval = 1
            reps = 0

        # Run SM-2
        new_ef, new_interval, new_reps = sm2_algorithm(quality, ef, interval, reps)
        next_review = datetime.now(timezone.utc) + timedelta(days=new_interval)

        if existing.data and len(existing.data) > 0:
            supabase.table("user_question_progress").update({
                "easiness_factor": new_ef,
                "interval_days": new_interval,
                "repetitions": new_reps,
                "next_review_date": next_review.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", user_id).eq("question_id", question_id).execute()
        else:
            supabase.table("user_question_progress").insert({
                "user_id": user_id,
                "question_id": question_id,
                "easiness_factor": new_ef,
                "interval_days": new_interval,
                "repetitions": new_reps,
                "next_review_date": next_review.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()

        return {
            "status": "ok",
            "new_ef": round(new_ef, 2),
            "new_interval": new_interval,
            "next_review": next_review.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/due-questions/{user_id}")
def get_due_questions(user_id: str):
    """Fetch questions that are due for review (next_review_date <= now)."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        progress_res = supabase.table("user_question_progress") \
            .select("*, quiz_questions(*)") \
            .eq("user_id", user_id) \
            .lte("next_review_date", now) \
            .execute()

        due = progress_res.data or []
        return {"status": "ok", "count": len(due), "questions": due}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Study Streak ───────────────────────────────────────────────

@app.get("/streak/{user_id}")
def get_streak(user_id: str):
    """Calculate the current study streak from quiz_attempts."""
    try:
        attempts_res = supabase.table("quiz_attempts") \
            .select("attempted_at") \
            .eq("user_id", user_id) \
            .order("attempted_at", desc=True) \
            .execute()

        attempts = attempts_res.data or []
        if not attempts:
            return {"streak": 0, "studied_today": False}

        # Get unique dates (in user's timezone — using UTC for simplicity)
        study_dates = set()
        for a in attempts:
            dt = datetime.fromisoformat(a["attempted_at"].replace("Z", "+00:00"))
            study_dates.add(dt.date())

        sorted_dates = sorted(study_dates, reverse=True)
        today = datetime.now(timezone.utc).date()

        # Check if studied today
        studied_today = sorted_dates[0] == today

        # Count consecutive days starting from today (or yesterday if not studied today yet)
        streak = 0
        check_date = today if studied_today else today - timedelta(days=1)

        for d in sorted_dates:
            if d == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            elif d < check_date:
                break

        return {"streak": streak, "studied_today": studied_today}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))