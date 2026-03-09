import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama

load_dotenv()

def gemini_pro_sql():
    """
    BEST FOR SQL: Uses Gemini 3.1 Pro.
    High reasoning capabilities for complex JOINs and schema mapping.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0,  # Critical for SQL to stay consistent
        max_output_tokens=1024, # Increased for complex SQL queries
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def gemini_flash_fast():
    """
    BEST FOR CHAT/TRANSLATION: Uses Gemini 3 Flash.
    Fast, cost-efficient, and perfect for simple intent/translation.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3, # Slight creativity for natural chat
        max_output_tokens=256,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# --- Legacy Support (Uncomment if needed) ---
# def openai_llm():
#     return ChatOpenAI(model="gpt-4o", temperature=0)

# def llama_llm():
#     return ChatOllama(model="qwen2.5-coder", base_url="http://localhost:11434
