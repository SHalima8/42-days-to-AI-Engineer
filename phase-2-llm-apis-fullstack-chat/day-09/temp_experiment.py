from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"  # swap to whatever has spare quota

thought = "I completely failed that presentation, I'm just bad at this and everyone noticed."

schema = {
    "type": "object",
    "properties": {
        "distortion_type": {"type": "string", "enum": ["all_or_nothing", "overgeneralization", "mind_reading", "catastrophizing", "labeling", "none"]},
        "confidence": {"type": "number"},
        "evidence_in_text": {"type": "string"},
        "reframed_thought": {"type": "string"},
        "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]}
    },
    "required": ["distortion_type", "confidence", "evidence_in_text", "reframed_thought", "severity"]
}

# Fixed for the whole experiment — only temperature changes
system_instruction = "You are a cognitive distortion detection engine."

def run_at_temperature(temperature, label):
    response = client.models.generate_content(
        model=MODEL,
        contents=thought,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature
        )
    )
    parsed = json.loads(response.text)
    print(f"--- {label} (temperature={temperature}) ---")
    print(json.dumps(parsed, indent=2))
    print()
    return parsed


# PART A — same temperature, same everything, run TWICE
# Tests: does low temperature actually produce near-identical output on repeats?
print("=== PART A: LOW TEMPERATURE, RUN TWICE (testing consistency) ===\n")
run_at_temperature(0.0, "LOW TEMP - RUN 1")
run_at_temperature(0.2, "LOW TEMP - RUN 2")

# PART B — same temperature (high), run TWICE
# Tests: does high temperature produce more variation between identical calls?
print("=== PART B: HIGH TEMPERATURE, RUN TWICE (testing variation) ===\n")
run_at_temperature(0.95, "HIGH TEMP - RUN 1")
run_at_temperature(1.0, "HIGH TEMP - RUN 2")