from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import uuid
import redis
import logging
from graph import build_graph
from llm import gemini_flash_fast  # Import the fast model for the translation route
from utils import translate_text
import vector_db
import os
from apscheduler.schedulers.background import BackgroundScheduler
from logging_config import setup_logging
from llm_cache import get_cached_answer, store_cached_answer
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Redis configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = os.getenv("SESSION_TYPE")

app.config["SESSION_REDIS"] = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT"))
)

app.config["SESSION_PERMANENT"] = os.getenv("SESSION_PERMANENT") == "True"
app.config["SESSION_USE_SIGNER"] = os.getenv("SESSION_USE_SIGNER") == "True"
app.config["SESSION_KEY_PREFIX"] = os.getenv("SESSION_KEY_PREFIX")
app.config["PERMANENT_SESSION_LIFETIME"] = int(os.getenv("SESSION_LIFETIME"))

Session(app)

logger.info("Redis session manager started.")

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
logger.info("Vector index background updater started.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chatbot endpoint.
    Handles concurrent users through Redis sessions.
    """

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        logger.info(f"[NEW SESSION] {session['session_id']}")

    payload = request.get_json()
    question = payload.get("message", "")

    logger.info(f"[USER QUESTION] {question}")
    
    # Check Redis LLM cache first
    cached = get_cached_answer(question)

    if cached:
        logger.info("LLM CACHE HIT")
        return jsonify(cached)

    logger.info("LLM CACHE MISS")
    
    logger.info(f"LLM CACHE HIT: {question}")
    logger.info(f"LLM CACHE MISS: {question}")
    
    # Pass thread_id via config, not in the input state
    config = {"configurable": {"thread_id": session["session_id"]}}

    result = graph.invoke(
        {"question": question},
        config=config
    )

    logger.info("[GRAPH EXECUTION COMPLETE]")

    response = {
        "answer": result.get("answer"),
        "graphBase64": result.get("graph_base64")
    }

    # store result
    store_cached_answer(question, response)

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
