from flask import Flask, request, jsonify, render_template
from graph import build_graph
from llm import llama_llm
from utils import translate_text

app = Flask(__name__)
llm = llama_llm()
graph = build_graph(llama_llm())

@app.route("/")
def index():
    return render_template("index.html")

conversation_history = [] # In-memory per session
@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    payload = request.get_json()
    question = payload.get("message", "")
    
    # Include conversation memory in the question
    context = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:] # Last 5 messages
    )
    question_with_context = f"{context}\nuser: {question}" if context else question
    
    result = graph.invoke({
        "question": question_with_context
    })
    
    # Save to memory
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "bot", "content": result["answer"]})
    
    response = {
        "answer": result.get("answer", "No data found."),
        "graphBase64": result.get("graph_base64")
    }
    
    return jsonify(response)
    
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
    
    translated, source_lang = translate_text(text, llm, target_lang=target_lang)    
    return jsonify({
        "translation": translated,
        "source_lang": source_lang
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)