import requests
import threading
import time
import random

BASE_URL = "http://127.0.0.1:5000"

# Chat messages
chat_messages = [
    "How many cooperatives are in the system?",
    "Give a bar graph of gender distribution in the cooperatives",
    "How many females and males are in the system",
    "How to register a cooperative?",
    "Which is the cooperative with the most members?",
    "Which cooperative has the least members?"
]

# Translation texts
translations = [
    {"text": "Hello world", "target_lang": "ar"},
    {"text": "كيف حالك؟", "target_lang": "en"},
    {"text": "Good morning", "target_lang": "ar"},
    {"text": "صباح الخير", "target_lang": "en"}
]

TOTAL_REQUESTS = 200
results = []

def send_chat_request(i):
    payload = {"message": random.choice(chat_messages)}
    try:
        start = time.time()
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        duration = time.time() - start
        results.append(duration)
        print(f"[CHAT] Request {i} | Status: {response.status_code} | Time: {duration:.2f}s")
    except Exception as e:
        print(f"[CHAT] Request {i} failed: {e}")

def send_translate_request(i):
    payload = random.choice(translations)
    try:
        start = time.time()
        response = requests.post(f"{BASE_URL}/translate", json=payload)
        duration = time.time() - start
        results.append(duration)
        print(f"[TRANSLATE] Request {i} | Status: {response.status_code} | Time: {duration:.2f}s")
    except Exception as e:
        print(f"[TRANSLATE] Request {i} failed: {e}")

threads = []

start_test = time.time()

for i in range(TOTAL_REQUESTS):
    # Randomly choose chat or translate
    if random.random() < 0.7:
        t = threading.Thread(target=send_chat_request, args=(i,))
    else:
        t = threading.Thread(target=send_translate_request, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

end_test = time.time()

print("\nTest Finished")
print("Total Requests:", TOTAL_REQUESTS)
print("Total Time:", round(end_test - start_test, 2), "seconds")
if results:
    print("Average Response Time:", round(sum(results)/len(results), 2), "seconds")