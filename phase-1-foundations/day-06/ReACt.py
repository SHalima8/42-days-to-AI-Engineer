from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt, label):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n{'='*60}")
    print(f"PROMPT TYPE: {label}")
    print(f"{'='*60}")
    print(f"RESPONSE:\n{response.choices[0].message.content}")
    return response.choices[0].message.content

# ============================================================
# DELIVERABLE 4 — ReAct Pattern (Reason + Act)
# Thought -> Action -> Observation -> Thought -> Action -> Answer
# ============================================================

react_prompt = """
You are an AI assistant that solves problems using tools.
You must follow this exact loop:
Thought: [what you know and what you need to find out]
Action: [which tool to use and what to ask it]
Observation: [what the tool returned]
... repeat as needed ...
Final Answer: [your conclusion]

Available tools:
- Calculator: performs math operations
- OrderDB: looks up order status by order ID
- WeatherAPI: returns current weather for a city

---
Question: A customer placed order #4521. 
They want to know if their outdoor delivery in Karachi will be affected today. 
The delivery fee was Rs.150 and they have a 10% discount. What is their final delivery fee?

Begin.
"""

react_no_pattern = f"""
A customer placed order #4521.
They want to know if their outdoor delivery in Karachi will be affected today.
The delivery fee was Rs.150 and they have a 10% discount. What is their final delivery fee?
"""

call_llm(react_no_pattern, "DELIVERABLE 4 - Without ReAct Pattern")
call_llm(react_prompt,     "DELIVERABLE 4 - With ReAct Pattern")