from flask import Flask, request, jsonify, render_template
from graph import build_graph
from llm import gemini_flash_fast  # Import the fast model for the translation route
from utils import translate_text
import vector_db
import os
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Initialize the fast model for the standalone translation route
llm_flash = gemini_flash_fast()

# The graph now handles its own internal LLM routing
graph = build_graph()

# Vector database to run in the background every 5 minutes for incremental updates
scheduler = BackgroundScheduler()

scheduler.add_job(
    vector_db.update_vector_index,
    "interval",
    minutes=5,
    max_instances=1
)

scheduler.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json()
    question = payload.get("message", "")
    
    result = graph.invoke({
        "question": question
    })
    
    response = {
        "answer": result.get("answer", "No data found."),
        "graphBase64": result.get("graph_base64")
    }
    
    return jsonify(response)

# health check
@app.route("/health")
def health():
    return {"status": "running"}
    
# Translation route
@app.route("/translate", methods=["POST"])
def translate():
    """
    Translates en to ar and vice versa
    """
    text = request.json.get("text")
    target_lang = request.json.get("target_lang")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Use the fast, cost-efficient model for standalone translations
    translated, source_lang = translate_text(text, llm_flash, target_lang=target_lang)    
    return jsonify({
        "translation": translated,
        "source_lang": source_lang
    })

if __name__ == "__main__":
    print("Loading vector database...")

    vector_db.get_vector_db()

    print("Vector DB ready.")
    app.run(host="0.0.0.0", debug=True, port=5000)
