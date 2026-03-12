import redis
import logging
from cachetools import TTLCache

# Logger for cache events
logger = logging.getLogger("cache")

# Redis connection
redis_client = redis.Redis(
    host="127.0.0.1", # force IPv4
    port=6379,
    decode_responses=True
)

# In-memory TTL cache (fast access layer)
# maxsize prevents memory leaks
# ttl ensures objects expire automatically
vector_cache = TTLCache(maxsize=500, ttl=600)
sql_cache = TTLCache(maxsize=200, ttl=600)

logger.info("Cache layer initialized.")