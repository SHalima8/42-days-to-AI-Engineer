"""
llm_client.py
Wraps Groq's chat completion API for the generation step.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from src import config

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

client = Groq(api_key=GROQ_API_KEY)


def generate_answer(augmented_prompt: str, temperature: float = None) -> dict:
    temperature = temperature if temperature is not None else config.TEMPERATURE

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded question-answering assistant. "
                        "Only answer using the provided context, which may include "
                        "both prose text and structured TABLE blocks. "
                        "If the context does not contain the answer, say so explicitly "
                        "instead of guessing."
                    ),
                },
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=temperature,
        )

        return {
            "answer": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    except Exception as e:
        return {"answer": None, "error": str(e)}