from flask import Flask, request, jsonify, render_template
from graph import build_graph
from llm import llama_llm
from langchain_core.messages import HumanMessage
from utils import detect_language, translate_text

app = Flask(__name__)
llm = llama_llm()
graph = build_graph(llama_llm())

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
    
    return jsonify({
        "reply": result["answer"]
    })
    
# Translation route
@app.route("/translate", methods=["POST"])
def translate():
    """
    Translates en to ar and vice versa
    """
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    translated, _ = translate_text(text, llm)
    
    return jsonify({"translation": translated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)