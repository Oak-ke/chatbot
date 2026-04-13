import hashlib
import json
from cache import redis_client
from datetime import date, datetime

CACHE_TTL = 3600  # 1 hour

def generate_cache_key(question: str) -> str:
    """
    Creates deterministic cache key for LLM queries
    
    What it does is user question -> redis cache check returns either Hit(return cahced answer)
    or Miss( run graph + store result)
    """
    qhash = hashlib.md5(question.strip().lower().encode()).hexdigest()
    return f"llm_cache:{qhash}"


def serialize_safe(obj):
    """
    Convert non-JSON-serializable objects.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    if isinstance(obj, dict):
        return {k: serialize_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize_safe(i) for i in obj]

    return obj

def get_cached_answer(question: str):
    key = generate_cache_key(question)
    cached = redis_client.get(key)

    if cached:
        return json.loads(cached)

    return None


def store_cached_answer(question: str, answer: dict):
    key = generate_cache_key(question)
    safe_answer = serialize_safe(answer)
    redis_client.setex(
        key,
        CACHE_TTL,
        json.dumps(safe_answer)
    )