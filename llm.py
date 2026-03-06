from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

_gemini = None

def openai_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        timeout=3,
        max_tokens=256, # Set a maximum number of tokens to prevent long responses
        request_timeout=5,
        top_p=1, # nucleus sampling to avoid rare words
        frequency_penalty=0.1,  # penalize repeated tokens
        presence_penalty=0.1  # penalize new topics
    )
    
def llama_llm():
    return ChatOllama(
        model="qwen2.5-coder",
        base_url="http://localhost:11434",
        max_tokens=256,
        top_p=1,
        num_ctx=512, # ensures full prompt is respected
        repeat_penalty=1.2 # prevents looping hallucinations
    )
    
def gemini_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=256
    )