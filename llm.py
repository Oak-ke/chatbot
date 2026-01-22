from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def openai_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        timeout=3
    )
    
def llama_llm():
    return ChatOllama(
        model="qwen2.5-coder",
        base_url="http://localhost:11434",
        temperature=0
    )