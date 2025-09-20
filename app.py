# ---------------------------
# 🔹 Imports
# ---------------------------
from flask import Flask, request, jsonify, render_template
import os
import spacy
import wikipedia
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from flask_cors import CORS  # Optional if frontend is served separately

# ---------------------------
# 🔹 Flask App Setup
# ---------------------------
app = Flask(__name__)
CORS(app)  # Only needed if using a separate frontend on another port

# ---------------------------
# 🔹 Configure Gemini API
# ---------------------------
genai.configure(api_key="AIzaSyCamUxck6FTXfWqB8wuVnPE9Atg4euECs8")  # <-- replace with your API key

# ---------------------------
# 🔹 Load Models
# ---------------------------
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-mpnet-base-v2")

# ---------------------------
# 🔹 Helper Functions
# ---------------------------

def extract_keywords_with_gemini(claim):
    """Use Gemini to extract keywords/entities from a claim."""
    prompt = f"""
    Extract the key entities, people, places, and concepts from the following statement to use for a search query.
    Return a list of keywords separated by commas.
    Statement: {claim}
    Keywords:"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        keywords = [k.strip() for k in response.text.split(",") if k.strip()]
        return keywords
    except Exception as e:
        print(f"Error extracting keywords with Gemini: {e}")
        return []

def fetch_wikipedia_summary(query):
    """Fetch Wikipedia summary and URL."""
    try:
        page = wikipedia.page(query, auto_suggest=False)
        text = wikipedia.summary(query, sentences=3, auto_suggest=False)
        url = page.url
        return {"text": text, "url": url}
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            choice = e.options[0]
            page = wikipedia.page(choice, auto_suggest=False)
            text = wikipedia.summary(choice, sentences=3, auto_suggest=False)
            url = page.url
            return {"text": text, "url": url}
        except:
            return {"text": "", "url": ""}
    except:
        return {"text": "", "url": ""}

def build_index(snippets):
    """Build FAISS index from text snippets."""
    filtered = [s for s in snippets if s["text"] and len(s["text"]) > 20]
    if not filtered:
        raise ValueError("No valid snippets to build an index.")
    texts = [s["text"] for s in filtered]
    urls = [s["url"] for s in filtered]
    embs = embedder.encode(texts, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, embs, texts, urls

def semantic_search(query, index, texts, urls, k=5):
    """Perform semantic search on the FAISS index."""
    q_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
    if np.linalg.norm(q_emb) != 0:
        faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if score > 0:
            results.append({"sim": float(score), "text": texts[idx], "url": urls[idx]})
    return results

def gemini_verdict(claim, evidences):
    """Use Gemini to give a verdict on the claim."""
    evidence_text = "\n".join([f"- {e['text'][:300]}..." for e in evidences])
    prompt = f"""
Claim: {claim}
Evidence:
{evidence_text}

Based on the provided evidence, classify the claim as TRUE, FALSE, or UNVERIFIABLE.
Provide a concise explanation and a confidence score (0-100).
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# 🔹 Flask Routes
# ---------------------------

@app.route("/")
def home():
    """Serve the frontend HTML page."""
    return render_template("index.html")

@app.route("/factcheck", methods=["POST"])
def factcheck():
    """Fact-checking API endpoint."""
    data = request.get_json()
    claim = data.get("claim", "")
    if not claim:
        return jsonify({"error": "No claim provided"}), 400

    try:
        # Extract keywords/entities
        keywords = extract_keywords_with_gemini(claim)
        snippets = [fetch_wikipedia_summary(k) for k in keywords]

        if not any(s["text"] for s in snippets):
            return jsonify({
                "claim": claim,
                "verdict": "UNVERIFIABLE. No evidence found.",
                "evidence": []
            })

        # Build FAISS index & search
        index, embs, texts, urls = build_index(snippets)
        results = semantic_search(claim, index, texts, urls, k=5)
        relevant_results = [r for r in results if r['sim'] > 0.5]

        # Generate verdict using Gemini
        if not relevant_results:
            verdict = "UNVERIFIABLE. No strong evidence found in retrieved articles."
        else:
            verdict = gemini_verdict(claim, relevant_results)

        return jsonify({
            "claim": claim,
            "verdict": verdict,
            "evidence": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 🔹 Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Change port if needed
