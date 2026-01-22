from flask import Flask, request, jsonify, render_template
from graph import build_graph
from llm import llama_llm
from langchain_core.messages import HumanMessage

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
    
    prompt = (
        "Translate the following text to Arabic if it is English, "
        "or to English if it is Arabic. "
        "Return ONLY the translation.\n\n"
        f"Text:\n{text}"
    )
    
    result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return jsonify({"translation": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)