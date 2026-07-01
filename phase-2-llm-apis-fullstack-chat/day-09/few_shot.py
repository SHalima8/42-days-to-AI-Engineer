from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3-flash-preview"  # swap to whichever model has spare quota

new_thought = "My partner didn't text back for two hours, they must be losing interest in me."

weak_system_instruction = "You are a helpful assistant that analyzes thoughts for cognitive distortions."


# ============================================================
# VERSION A — ZERO-SHOT (casual prompt, no examples, no schema)
# ============================================================
zero_shot_prompt = (
    f'Analyze this thought and just give me the result as JSON: "{new_thought}" '
    f'Include distortion type, confidence, the evidence, a reframe, and severity.'
)

response_zero_shot = client.models.generate_content(
    model=MODEL,
    contents=zero_shot_prompt,
    config=types.GenerateContentConfig(
        system_instruction=weak_system_instruction
    )
)

print("=" * 60)
print("VERSION A — ZERO-SHOT (RAW OUTPUT)")
print("=" * 60)
print(response_zero_shot.text)

print("\n--- Attempting to parse Zero-Shot as JSON ---")
try:
    parsed_zero = json.loads(response_zero_shot.text)
    print("Parsed successfully:")
    print(json.dumps(parsed_zero, indent=2))
except json.JSONDecodeError as e:
    print(f"FAILED TO PARSE: {e}")


# ============================================================
# VERSION B — FEW-SHOT (same casual instruction style, but
# preceded by 2 worked examples showing the exact JSON shape)
# ============================================================
few_shot_contents = [
    {"role": "user", "parts": [{"text": 'Analyze this thought and just give me the result as JSON: "I completely failed that presentation, I\'m just bad at this and everyone noticed." Include distortion type, confidence, the evidence, a reframe, and severity.'}]},
    {"role": "model", "parts": [{"text": json.dumps({
        "distortion_type": "labeling",
        "confidence": 0.9,
        "evidence_in_text": "I'm just bad at this",
        "reframed_thought": "This specific presentation was difficult, but one performance doesn't define my overall abilities.",
        "severity": "moderate"
    })}]},

    {"role": "user", "parts": [{"text": 'Analyze this thought and just give me the result as JSON: "If I don\'t get this promotion, my whole career is basically over." Include distortion type, confidence, the evidence, a reframe, and severity.'}]},
    {"role": "model", "parts": [{"text": json.dumps({
        "distortion_type": "catastrophizing",
        "confidence": 0.92,
        "evidence_in_text": "my whole career is basically over",
        "reframed_thought": "Missing this promotion would be disappointing, but it's one step in a longer career, not the end of it.",
        "severity": "moderate"
    })}]},

    # The real question — same casual style, new thought, zero prior exposure to this specific example
    {"role": "user", "parts": [{"text": f'Analyze this thought and just give me the result as JSON: "{new_thought}" Include distortion type, confidence, the evidence, a reframe, and severity.'}]},
]

response_few_shot = client.models.generate_content(
    model=MODEL,
    contents=few_shot_contents,
    config=types.GenerateContentConfig(
        system_instruction=weak_system_instruction
    )
)

print("\n" + "=" * 60)
print("VERSION B — FEW-SHOT (RAW OUTPUT)")
print("=" * 60)
print(response_few_shot.text)

print("\n--- Attempting to parse Few-Shot as JSON ---")
try:
    parsed_few = json.loads(response_few_shot.text)
    print("Parsed successfully:")
    print(json.dumps(parsed_few, indent=2))
except json.JSONDecodeError as e:
    print(f"FAILED TO PARSE: {e}")


# ============================================================
# SIDE-BY-SIDE FIELD COMPARISON
# ============================================================
expected_fields = {"distortion_type", "confidence", "evidence_in_text", "reframed_thought", "severity"}

print("\n" + "=" * 60)
print("FIELD COMPLIANCE CHECK")
print("=" * 60)

        