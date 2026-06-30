from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = "Explain the water cycle in three sentences."

print("=== NON-STREAMING ===")
start = time.time()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
elapsed = time.time() - start
print(response.text)
print(f"\n(Full response received after {elapsed:.2f} seconds, all at once)")

print("\n=== STREAMING ===")
start = time.time()
first_chunk_time = None

for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=prompt
):
    if first_chunk_time is None:
        first_chunk_time = time.time() - start
    print(chunk.text, end="", flush=True)

total_elapsed = time.time() - start
print(f"\n\n(First chunk arrived after {first_chunk_time:.2f}s, full stream finished after {total_elapsed:.2f}s)")