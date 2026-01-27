import json
import re
import os
import logging
import pandas as pd
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage
from utils import detect_language, translate_text
from visualizer import Visualizer, FileDataSource, MockDataSource
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    language: str
    intent: str
    data: dict
    answer: str
    graph_url: str
    
def load_data():
    path = "data/public_data.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
    
DATA = load_data()

INTENT_MAP = {
    "system_name": ["system name", "name of system", "what is the system called"],
    "system_info": ["system_info", "system information", "about system", "info", "tell me more"],
    "cooperatives_total": ["cooperatives_total", "number of cooperatives", "cooperatives registered"],
    "members_total": ["members_total", "total members", "members", "female members", "male members"],
    "members_by_state": ["members_by_state", "members per state", "members by state"],
    "female_members": ["female members"],
    "male_members": ["male members"],
    "directors_total": ["directors_total", "number of directors", "directors"],
    "visualize": ["visualize", "graph", "plot", "chart", "show me trends"]
}

def detect_lan_and_translate(state: State, llm):
    text = state["question"]
    lang = detect_language(text)
        
    if lang == "en":
        return {
            "question": text,
            "language": "en",
        }
        
    translated, _ = translate_text(text, llm, target_lang="English")
    
    return {
        "question": translated,
        "language": "ar"
    }

def detect_intent(state: State, llm):
    prompt = (
        "Classify the intent into one of:\n"
        "- system_name\n"
        "- system_info\n"
        "- cooperatives_total\n"
        "- members_total\n"
        "- female_members\n"
        "- male_members\n"
        "- members_by_state\n"
        "- directors_total\n"
        "- visualize\n"
        "- unknown\n\n"
        f"Question: {state['question']}"
    )
    raw_intent = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    # Normalize to canonical intent using substring matching
    for canonical, aliases in INTENT_MAP.items():
        for alias in aliases:
            if alias.lower() in raw_intent:
                return {"intent": canonical}

    return {"intent": "unknown"}


def select_data(state: State):
    intent = state["intent"]
    
    if intent == "system_info":
        return {"data": DATA["system_description"]}
    
    if intent == "cooperatives_total":
        return {"data": DATA["cooperatives_registered"]}

    if intent == "directors_total":
        return {"data": DATA["directors"]}

    if intent == "members_total":
        return {"data": DATA["members"]}

    if intent == "members_by_state":
        return {"data": DATA["members_by_state"]}

    return {"data": None}

def generate_answer(state: State, llm):
    response = ""
    if state["data"] is None:
        response = "I can only provide general public information available in this system."
    else:
        prompt = (
            "Using the data below, answer the question clearly.\n"
            "Do NOT add new information, assumptions or extra facts.\n\n"
            f"Data:\n{state['data']}\n\n"
            f"Question:\n{state['question']}"
        )
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    if state.get("graph_url"):
        if response and response != "I can only provide general public information available in this system.":
            response += "\n\n"
        else:
            response = "" # Clear the "I can only provide..." message if we have a real chart? 
            # Actually, user said: The agent should be able to say, "I've analyzed the current data state and generated this visualization."
        
        response += f"I've analyzed the current data state and generated this visualization: {state['graph_url']}"
        
    return {"answer": response}



def visualize_node(state: State):
    temp_path = None
    # Determine strategy
    if os.path.exists("data/public_data.json"):
        # We need to transform the JSON structure into a flat DF for visualization if possible
        with open("data/public_data.json", "r") as f:
            raw_data = json.load(f)
        
        # If it's the members_by_state, it's easy to visualize
        if state["intent"] == "members_by_state":
            df = pd.DataFrame(list(raw_data["members_by_state"].items()), columns=["State", "Members"])
            # Temporary file strategy for this specific DF
            temp_path = f"data/temp_{uuid.uuid4().hex}.csv"
            logger.info(f"Creating temporary CSV at {temp_path} for intent '{state['intent']}'. Data shape: {df.shape}")
            df.to_csv(temp_path, index=False)
            strategy = FileDataSource(temp_path)
        else:
            strategy = MockDataSource()
    else:
        strategy = MockDataSource()
        
    try:
        viz = Visualizer(strategy)
        intent = state.get("intent", "unknown")
        safe_intent = re.sub(r'[^a-zA-Z0-9_]', '_', intent)
        output_filename = f"static/graphs/viz_{safe_intent}.png"
        graph_path = viz.analyze_and_plot(output_path=output_filename)
        
        if not graph_path:
            return {"graph_url": None}
        
        # In a web app, we want the URL relative to static
        graph_url = f"/static/graphs/{os.path.basename(graph_path)}"
        
        return {"graph_url": graph_url}
    finally:
        if temp_path and os.path.exists(temp_path):
            logger.info(f"Deleting temporary CSV at {temp_path}")
            os.remove(temp_path)

def route_to_answer(state: State):
    if state["intent"] == "visualize" or state["intent"] == "members_by_state":
        return "visualize"
    return "answer"


def build_graph(llm):
    graph = StateGraph(State)

    graph.add_node("translate", lambda s: detect_lan_and_translate(s, llm))
    graph.add_node("intent", lambda s: detect_intent(s, llm))
    graph.add_node("data", select_data)
    graph.add_node("visualize", visualize_node)
    graph.add_node("answer", lambda s: generate_answer(s, llm))

    graph.set_entry_point("translate")
    graph.add_edge("translate", "intent")
    graph.add_edge("intent", "data")
    
    graph.add_conditional_edges(
        "data",
        route_to_answer,
        {
            "visualize": "visualize",
            "answer": "answer"
        }
    )
    graph.add_edge("visualize", "answer")

    return graph.compile()