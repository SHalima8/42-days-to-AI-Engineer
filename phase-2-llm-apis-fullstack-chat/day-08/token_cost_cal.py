from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini 2.5 Flash-Lite pricing (per 1 million tokens)
INPUT_PRICE_PER_MILLION = 0.10
OUTPUT_PRICE_PER_MILLION = 0.40

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain what an API is in one sentence, simply."
)

usage = response.usage_metadata
input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
total_cost = input_cost + output_cost

print("=== Token metrics & cost calculation ===")

print(f"Input tokens:  {input_tokens}  -> ${input_cost:.8f}")
print(f"Output tokens: {output_tokens}  -> ${output_cost:.8f}")
print(f"Total tokens:  {total_tokens}  -> ${total_cost:.8f}")