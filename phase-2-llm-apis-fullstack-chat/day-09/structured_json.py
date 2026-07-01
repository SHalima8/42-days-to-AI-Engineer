from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

thought = "I completely failed that presentation, I'm just bad at this and everyone noticed."

MODEL = "gemini-3-flash-preview" 

# ============================================================
# VERSION A — Prompted JSON, no schema enforcement
# ============================================================
# ============================================================
# VERSION A — Casual/weak prompting, JSON requested informally
# in the USER message, no strict system instruction
# ============================================================
weak_system_instruction = "You are a helpful assistant that analyzes thoughts for cognitive distortions."

weak_user_prompt = (
    f'Analyze this thought and just give me the result as JSON: "{thought}" '
    f'Include distortion type, confidence, the evidence, a reframe, and severity.'
)

response_a = client.models.generate_content(
    model=MODEL,
    contents=weak_user_prompt,
    config=types.GenerateContentConfig(
        system_instruction=weak_system_instruction
        # still no response_mime_type, no response_schema
    )
)

print("=" * 60)
print("VERSION A — CASUAL JSON REQUEST (RAW OUTPUT)")
print("=" * 60)
print(response_a.text)

print("\n--- Attempting to parse Version A as JSON ---")
try:
    parsed_a = json.loads(response_a.text)
    print("Parsed successfully:")
    print(json.dumps(parsed_a, indent=2))
except json.JSONDecodeError as e:
    print(f"FAILED TO PARSE: {e}")
    print("(This is the exact failure mode schema enforcement is meant to prevent.)")


# ============================================================
# VERSION B — Schema-enforced JSON (constrained decoding)
# ============================================================
distortion_schema = {
    "type": "object",
    "properties": {
        "distortion_type": {
            "type": "string",
            "enum": ["all_or_nothing", "overgeneralization", "mind_reading", "catastrophizing", "labeling", "none"]
        },
        "confidence": {"type": "number"},
        "evidence_in_text": {"type": "string"},
        "reframed_thought": {"type": "string"},
        "severity": {
            "type": "string",
            "enum": ["mild", "moderate", "severe"]
        }
    },
    "required": ["distortion_type", "confidence", "evidence_in_text", "reframed_thought", "severity"]
}

schema_instruction = (
    "You are a cognitive distortion detection engine. "
    "Analyze the user's thought and identify the primary cognitive distortion present, if any."
)

response_b = client.models.generate_content(
    model=MODEL,
    contents=thought,
    config=types.GenerateContentConfig(
        system_instruction=schema_instruction,
        response_mime_type="application/json",
        response_schema=distortion_schema
    )
)

print("\n" + "=" * 60)
print("VERSION B — SCHEMA-ENFORCED JSON (RAW OUTPUT)")
print("=" * 60)
print(response_b.text)

print("\n--- Attempting to parse Version B as JSON ---")
parsed_b = json.loads(response_b.text)
print("Parsed successfully:")
print(json.dumps(parsed_b, indent=2))