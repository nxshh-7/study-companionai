import os
import json
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()  # reads your .env file

app = Flask(__name__)

# ---------- Groq setup ----------
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_KEY = os.getenv("GROQ_API_KEY")

if GROQ_KEY:
    print("Groq key found:", GROQ_KEY[:4] + "...")
    groq_client = Groq(api_key=GROQ_KEY)
else:
    print("ERROR: GROQ_API_KEY not found. Check your .env file!")
    groq_client = None


def ask_ai(system, user):
    if not groq_client:
        raise Exception("No Groq key found. Put GROQ_API_KEY in your .env file.")

    last_error = None
    for attempt in range(2):  # try twice
        try:
            reply = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return reply.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"[Groq] failed: {str(e)[:200]}")
            time.sleep(2)
    raise last_error


# ---------- material handling ----------
store = {"chunks": [], "vectorizer": None, "matrix": None}


def split_text(text, page, size=800):
    return [{"text": text[i:i + size], "page": page}
            for i in range(0, len(text), size) if text[i:i + size].strip()]


def get_material(limit=12000):
    return "\n".join(c["text"] for c in store["chunks"])[:limit]


def clean_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise Exception("The AI did not return a list. Please click the button again.")
    return json.loads(text[start:end + 1])


# ---------- routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    chunks = []
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(file)
        for number, page in enumerate(reader.pages, start=1):
            chunks += split_text(page.extract_text() or "", number)
    else:
        chunks = split_text(file.read().decode("utf-8", errors="ignore"), 1)

    if not chunks:
        return jsonify({"error": "Could not read any text from this file."}), 400

    store["chunks"] = chunks
    store["vectorizer"] = TfidfVectorizer(stop_words="english")
    store["matrix"] = store["vectorizer"].fit_transform([c["text"] for c in chunks])
    return jsonify({"message": f"Uploaded! Found {len(chunks)} chunks."})


@app.route("/ask", methods=["POST"])
def ask():
    if not store["chunks"]:
        return jsonify({"error": "Upload a file first."}), 400
    question = request.json["question"]

    q_vec = store["vectorizer"].transform([question])
    scores = cosine_similarity(q_vec, store["matrix"])[0]
    top = scores.argsort()[-4:][::-1]
    best = [store["chunks"][i] for i in top if scores[i] > 0]

    if not best:
        return jsonify({"answer": "I couldn't find this in your material.", "sources": []})

    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in best)
    system = ("You are a study helper. Answer ONLY using the material given. "
              "If the answer is not in the material, say exactly: "
              "'I couldn't find this in your material.' Explain simply.")
    answer = ask_ai(system, f"MATERIAL:\n{context}\n\nQUESTION: {question}")
    pages = sorted({c["page"] for c in best})
    return jsonify({"answer": answer, "sources": pages})


@app.route("/summary", methods=["POST"])
def summary():
    if not store["chunks"]:
        return jsonify({"error": "Upload a file first."}), 400
    system = "Summarize ONLY the given material in simple bullet points."
    return jsonify({"result": ask_ai(system, get_material())})


@app.route("/quiz", methods=["POST"])
def quiz():
    if not store["chunks"]:
        return jsonify({"error": "Upload a file first."}), 400
    system = ("Create 5 multiple choice questions ONLY from the given material. "
              "Reply with ONLY JSON, no other text, in this format: "
              '[{"question": "...", "options": ["A...", "B...", "C...", "D..."], '
              '"answer": "A", "explanation": "..."}]')
    return jsonify({"result": clean_json(ask_ai(system, get_material()))})


@app.route("/flashcards", methods=["POST"])
def flashcards():
    if not store["chunks"]:
        return jsonify({"error": "Upload a file first."}), 400
    system = ("Create 8 flashcards ONLY from the given material. "
              "Reply with ONLY JSON, no other text, in this format: "
              '[{"front": "question or term", "back": "answer"}]')
    return jsonify({"result": clean_json(ask_ai(system, get_material()))})


@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)