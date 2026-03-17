import hashlib
import json
from cache import redis_client

CACHE_TTL = 3600  # 1 hour

def generate_cache_key(question: str) -> str:
    """
    Creates deterministic cache key for LLM queries
    
    What it does is user question -> redis cache check returns either Hit(return cahced answer)
    or Miss( run graph + store result)
    """
    qhash = hashlib.md5(question.strip().lower().encode()).hexdigest()
    return f"llm_cache:{qhash}"


def get_cached_answer(question: str):
    key = generate_cache_key(question)
    cached = redis_client.get(key)

    if cached:
        return json.loads(cached)

    return None


def store_cached_answer(question: str, answer: dict):
    key = generate_cache_key(question)
    redis_client.setex(
        key,
        CACHE_TTL,
        json.dumps(answer)
    )