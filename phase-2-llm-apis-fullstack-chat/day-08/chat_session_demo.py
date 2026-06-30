from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== METHOD 1: generate_content (stateless) ===")
response1 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="My name is Sadia."
)
print("Response 1:", response1.text)

response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is my name?"
)
print("Response 2:", response2.text)
print("(Notice: the model has no memory of the first message)")

print("\n=== METHOD 2: chat session (stateful) ===")
chat = client.chats.create(model="gemini-2.5-flash")

response3 = chat.send_message("My name is Sadia.")
print("Response 1:", response3.text)

response4 = chat.send_message("What is my name?")
print("Response 2:", response4.text)
print("(Notice: the chat session remembered the name)")