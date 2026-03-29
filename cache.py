import redis
import logging
from cachetools import TTLCache
import os
from logging_config import setup_logging

# Logger for cache events
setup_logging()
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Redis connection
redis_client = redis.Redis(
    host=REDIS_HOST, # force IPv4
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=2, # Prevents redis from stalling
    socket_timeout=2
)

# In-memory TTL cache (fast access layer)
# maxsize prevents memory leaks
# ttl ensures objects expire automatically
vector_cache = TTLCache(maxsize=500, ttl=600)
sql_cache = TTLCache(maxsize=200, ttl=600)

logger.info("Cache layer initialized.")