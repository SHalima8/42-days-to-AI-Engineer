from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3-flash-preview"  # swap to whichever model has spare quota

def structured_distortion_detection(thought: str):
    schema = {
        "type": "object",
        "properties": {
            "distortion_type": {
                "type": "string",
                "enum": ["all_or_nothing", "overgeneralization", "mind_reading", "catastrophizing", "labeling", "none"]
            },
            "confidence": {"type": "number"},
            "evidence_in_text": {"type": "string"},
            "reframed_thought": {"type": "string"},
            "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]}
        },
        "required": ["distortion_type", "confidence", "evidence_in_text", "reframed_thought", "severity"]
    }

    response = client.models.generate_content(
        model=MODEL,
        contents=thought,
        config=types.GenerateContentConfig(
            system_instruction="You are a cognitive distortion detection engine.",
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2  # low — we want consistent classification, not creative variation
        )
    )
    return json.loads(response.text)    

def parse_messy_journal_entry(entry: str):
    system_instruction = (
        "You are a reflective journaling assistant. Read the entry and identify: "
        "1) the main emotional theme, 2) any recurring worries, and 3) any cognitive distortions "
        "you notice, described in plain language. Do not use JSON — write a short, natural reflection."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=entry,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.6  # moderate — some interpretive freedom is appropriate here
        )
    )
    return response.text


messy_entry = (
    "idk today was just weird, work was fine i guess but then sarah didn't say bye when she left "
    "and now i cant stop thinking she's annoyed with me, and honestly this happens every week, "
    "someone gets quiet and i just assume its me, i probably do this too much lol, also didnt sleep well"
)

def generate_validation_function():
    system_instruction = (
        "You are a senior Python engineer. Write clean, well-commented Python code. "
        "Only output the code, no explanation outside of code comments."
    )

    user_prompt = (
        "Write a Python function called `validate_distortion_output(data: dict) -> bool` "
        "that checks whether a dictionary matches this schema: "
        "distortion_type (must be one of: all_or_nothing, overgeneralization, mind_reading, "
        "catastrophizing, labeling, none), confidence (float between 0 and 1), "
        "evidence_in_text (non-empty string), reframed_thought (non-empty string), "
        "severity (must be one of: mild, moderate, severe). "
        "Return True if valid, False otherwise, and print a specific reason when invalid."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3  # low — correctness matters more than stylistic variety
        )
    )
    return response.text

def summarize_reflection(long_entry: str):
    system_instruction = (
        "You are a concise clinical-style summarizer. Summarize the user's reflection in 2-3 sentences, "
        "capturing the emotional core and any pattern of thinking, without adding advice or interpretation "
        "beyond what's stated."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=long_entry,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4  # low-moderate — factual condensation, minimal creative drift
        )
    )
    return response.text


long_reflection = (
    "This week has honestly been a lot. Monday I had that presentation and I really felt like I bombed it, "
    "even though my manager said it went fine. Then Wednesday my partner and I had a small disagreement about "
    "weekend plans, and I spent way too long afterward replaying it, convinced I'd said something wrong. "
    "By Friday I was just exhausted, not from doing much, but from all the second-guessing. I keep noticing "
    "I do this a lot — assume the worst about how people see me, then spend hours checking it in my head."
)


if __name__ == "__main__":
    print("=== PROMPT 1: Structured JSON Generation ===")
    print(json.dumps(structured_distortion_detection(
        "I completely failed that presentation, I'm just bad at this."
    ), indent=2))

    print("\n=== PROMPT 2: Unstructured Text Parsing ===")
    print(parse_messy_journal_entry(messy_entry))

    print("\n=== PROMPT 3: Code Generation ===")
    print(generate_validation_function())

    print("\n=== PROMPT 4: Document Summarization ===")
    print(summarize_reflection(long_reflection))    