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

# Custom task: classify customer complaint URGENCY + DEPARTMENT
# The model has no idea what your urgency scale or departments are
# without examples

test_complaint = "I ordered 3 items but only received 1. I need the rest before my event tomorrow."

# ZERO SHOT — model guesses your format
zero_shot = f"""
Classify this customer complaint.

Complaint: "{test_complaint}"
"""

# ONE SHOT — model sees format once
one_shot = f"""
Classify customer complaints into urgency and department.

Example:
Complaint: "My payment was charged twice."
Urgency: HIGH
Department: BILLING
Action: Refund duplicate charge immediately

Now classify:
Complaint: "{test_complaint}"
"""

# FEW SHOT — model sees multiple situations
few_shot = f"""
Classify customer complaints into urgency and department.
Urgency levels: LOW, MEDIUM, HIGH, CRITICAL
Departments: BILLING, LOGISTICS, TECHNICAL, RETURNS

Example 1:
Complaint: "My payment was charged twice."
Urgency: HIGH
Department: BILLING
Action: Refund duplicate charge immediately

Example 2:
Complaint: "The color of the product is slightly different from the photo."
Urgency: LOW
Department: RETURNS
Action: Log feedback, offer return if requested

Example 3:
Complaint: "Website keeps crashing when I try to checkout."
Urgency: CRITICAL
Department: TECHNICAL
Action: Escalate to dev team, offer manual order processing

Example 4:
Complaint: "My order has been stuck in Lahore warehouse for 5 days."
Urgency: HIGH
Department: LOGISTICS
Action: Investigate shipment delay, provide update within 2 hours

Now classify:
Complaint: "{test_complaint}"
"""

call_llm(zero_shot, "ZERO-SHOT — No examples, model guesses format")
call_llm(one_shot,  "ONE-SHOT — One example, model learns basic format")
call_llm(few_shot,  "FEW-SHOT — Four examples, model handles edge cases")