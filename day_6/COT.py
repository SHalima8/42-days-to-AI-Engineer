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

# Problem 1 — Math word problem
problem1 = "A shop sells 3 types of boxes. Small costs Rs.150, Medium costs Rs.300, Large costs Rs.500. Ali buys 4 small, 2 medium, and 1 large. He pays with Rs.2000. How much change does he get?"

# Problem 2 — Logic problem  
problem2 = "There are 3 friends: Ayesha, Bilal, and Chand. Ayesha is older than Bilal. Chand is younger than Bilal. Who is the oldest and who is the youngest?"

# WITHOUT COT
call_llm(f"Answer this: {problem1}", 
         "PROBLEM 1 — Direct (No CoT)")

# WITH COT
call_llm(f"Answer this step by step: {problem1}\nLet's think step by step.", 
         "PROBLEM 1 — Chain of Thought")

# WITHOUT COT
call_llm(f"Answer this: {problem2}", 
         "PROBLEM 2 — Direct (No CoT)")

# WITH COT
call_llm(f"Answer this step by step: {problem2}\nLet's think step by step.", 
         "PROBLEM 2 — Chain of Thought")