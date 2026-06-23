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
    print(f"PROMPT:\n{prompt}")
    print(f"\nRESPONSE:\n{response.choices[0].message.content}")
    return response.choices[0].message.content


anatomy_prompt = """
ROLE: You are a sentiment analysis expert trained on customer feedback data.

CONTEXT: A Pakistani e-commerce company receives customer reviews in English.
They need to classify each review to improve their product quality.

TASK: Analyze the sentiment of the customer review provided below.
Determine whether it is Positive, Negative, or Neutral.

EXAMPLES:
Review: "The product arrived on time and works perfectly!"
Sentiment: Positive

Review: "Completely broken on arrival. Very disappointed."
Sentiment: Negative

OUTPUT FORMAT:
Sentiment: [Positive/Negative/Neutral]
Confidence: [High/Medium/Low]
Reason: [One sentence explanation]

---
Review to analyze: "The quality is okay but delivery took forever." and this "It is a very bad product. I want my money back."
"""

call_llm(anatomy_prompt, "DELIVERABLE 1 - Full Prompt Anatomy (All 5 Components)")



