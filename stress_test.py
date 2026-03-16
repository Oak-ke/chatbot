import requests
import threading
import time
import random

URL = "http://127.0.0.1:5000"

messages = [
    "How many cooperatives are in the system?",
    "Give a bar graph of gender distribution in the cooperatives",
    "How many females and males are in the system",
    "How to register a cooperative?",
    "Which is the cooperative with the most members?",
    "Which cooperative has the least members?"
]

payload = {
    "message": random.choice(messages)
}

TOTAL_REQUESTS = 20000
results = []

def send_request(i):
    payload = {
        "message": random.choice(messages)
    }
    
    try:
        start = time.time()

        response = requests.post(URL, json=payload)

        duration = time.time() - start

        results.append(duration)

        print(f"Request {i} | Status: {response.status_code} | Time: {duration:.2f}s")

    except Exception as e:
        print(f"Request {i} failed: {e}")


threads = []

start_test = time.time()

for i in range(TOTAL_REQUESTS):
    t = threading.Thread(target=send_request, args=(i,))
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