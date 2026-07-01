from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def call_gemini(contents, system_instruction, label):
    """Reusable call with retry-on-failure, used across all test cases below."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            print(f"--- {label} ---")
            print(response.text)
            print()
            return response.text
        except Exception as e:
            print(f"[{label}] Attempt {attempt + 1} failed: {e}")
            time.sleep(2)


question = "Explain what an API is in one sentence, simply."

blunt_persona = "You are a blunt, no-nonsense senior engineer. You never use analogies or metaphors — only precise technical language."
teacher_persona = "You are a warm, patient teacher explaining things to a curious 10-year-old. Use simple analogies."


# CASE 1: Same question, two different system instructions
# Demonstrates: system instruction alone shapes tone/vocabulary, no history needed
call_gemini(
    contents=question,
    system_instruction=blunt_persona,
    label="CASE 1a — BLUNT ENGINEER (system instruction only)"
)

call_gemini(
    contents=question,
    system_instruction=teacher_persona,
    label="CASE 1b — PATIENT TEACHER (system instruction only)"
)


# CASE 2: Follow-up question with NO history
# Demonstrates: without prior turns, "it" refers to nothing — model hallucinates a topic
call_gemini(
    contents=[
        {"role": "user", "parts": [{"text": "Now explain it again, but for a 10-year-old."}]}
    ],
    system_instruction="You are a helpful teacher.",
    label="CASE 2 — FOLLOW-UP WITH NO HISTORY (broken — 'it' has no referent)"
)


# CASE 3: Same follow-up question, WITH real history
# Demonstrates: correct 3-role structure — user, model, user — 'it' now resolves correctly
call_gemini(
    contents=[
        {"role": "user", "parts": [{"text": "Explain what an API is in one sentence, simply."}]},
        {"role": "model", "parts": [{"text": "An API is a set of definitions and protocols that allows different software applications to communicate with each other."}]},
        {"role": "user", "parts": [{"text": "Now explain it again, but for a 10-year-old."}]}
    ],
    system_instruction="You are a helpful teacher.",
    label="CASE 3 — FOLLOW-UP WITH HISTORY (correct — 'it' resolves to API)"
)