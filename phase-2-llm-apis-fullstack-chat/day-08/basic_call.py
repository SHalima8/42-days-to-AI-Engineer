from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize the client with your API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Send a structured prompt
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain what an API is in one sentence, simply."
)

# Print the generated text
print("--- RESPONSE TEXT ---")
print(response.text)

# Print the full raw response object so we can see its actual structure
print("\n--- FULL RAW RESPONSE OBJECT ---")
print(response)