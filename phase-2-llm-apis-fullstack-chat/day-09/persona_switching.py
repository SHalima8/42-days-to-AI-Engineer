from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"  # swap to whatever has spare quota

thought = "I completely failed that presentation, I'm just bad at this and everyone noticed."

# The 3 personas — only difference between calls
PERSONAS = {
    "formal": (
        "You are a clinical CBT-trained cognitive distortion analyst. "
        "Respond in precise, professional, evidence-based language. Explain what distortion "
        "you notice and offer a reframe, in a few sentences of natural prose. No casual phrasing."
    ),
    "casual": (
        "You're a friendly, down-to-earth mental wellness buddy. Respond like a supportive friend "
        "would in conversation — relaxed, warm, no jargon. Point out the thinking pattern gently "
        "and offer a reframe, in a few natural sentences."
    ),
    "technical": (
        "You are a cognitive-behavioral pattern classification system. Describe what you notice "
        "using precise psychological terminology, as if writing for someone familiar with CBT theory. "
        "Respond in a few sentences of natural prose, not a list."
    ),
}

def call_with_persona(persona_key: str, contents):
    if persona_key not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona_key}'. Choose from: {list(PERSONAS.keys())}")

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=PERSONAS[persona_key],
            temperature=0.7
            # no response_mime_type, no response_schema — free-form text this time
        )
    )
    print(f"--- PERSONA: {persona_key.upper()} ---")
    print(response.text)
    print()
    return response.text


# ============================================================
# TEST 1 — Same question, all 3 personas, run independently
# ============================================================
print("=== TEST 1: PERSONA SWITCH, PLAIN CONVERSATIONAL OUTPUT ===\n")
call_with_persona("formal", thought)
call_with_persona("casual", thought)
call_with_persona("technical", thought)