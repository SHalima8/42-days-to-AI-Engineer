# day-11/gemini_client.py
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.0-flash-lite"  # match whatever model you used in Day 8


def generate_reply(history: list[dict]) -> dict:
    system_instruction = next(
        (turn["content"] for turn in history if turn["role"] == "system"),
        None,
    )

    contents = [
        {"role": "user" if turn["role"] == "user" else "model",
         "parts": [{"text": turn["content"]}]}
        for turn in history
        if turn["role"] != "system"
    ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config={"system_instruction": system_instruction} if system_instruction else None,
    )

    return {
        "text": response.text,
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }