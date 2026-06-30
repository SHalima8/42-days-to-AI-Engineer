from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compare(bad_prompt, good_prompt, practice_num, practice_name):
    print(f"\n{'='*60}")
    print(f"BEST PRACTICE #{practice_num}: {practice_name}")
    print(f"{'='*60}")
    
    bad_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": bad_prompt}]
    ).choices[0].message.content

    good_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": good_prompt}]
    ).choices[0].message.content

    print(f"\n❌ BAD PROMPT:\n{bad_prompt}")
    print(f"\n❌ BAD RESPONSE:\n{bad_response}")
    print(f"\n✅ GOOD PROMPT:\n{good_prompt}")
    print(f"\n✅ GOOD RESPONSE:\n{good_response}")

# ============================================================
# 10 BEST PRACTICES — Before/After
# ============================================================

compare(
    bad_prompt="Tell me about AI.",
    good_prompt="Explain what AI is in 3 bullet points, each under 15 words, for a non-technical audience.",
    practice_num=1,
    practice_name="Be specific about length and format"
)

compare(
    bad_prompt="Summarize this.",
    good_prompt="Summarize the following text in 2 sentences focusing on the main finding:\n'Researchers at MIT found that daily 10-minute walks reduce anxiety by 40% in adults over 50.'",
    practice_num=2,
    practice_name="Always include the actual content to process"
)

compare(
    bad_prompt="Write a professional email.",
    good_prompt="Write a professional email from an intern to their manager requesting a 1-on-1 meeting. Tone: polite and brief. Length: under 80 words.",
    practice_num=3,
    practice_name="Specify role, tone, and length"
)

compare(
    bad_prompt="Is this review good or bad? 'Product is okay but late delivery.'",
    good_prompt="Classify this review as Positive, Negative, or Neutral. Reply with one word only.\nReview: 'Product is okay but late delivery.'",
    practice_num=4,
    practice_name="Constrain output to exactly what you need"
)

compare(
    bad_prompt="Translate: 'The meeting is at 3pm tomorrow'",
    good_prompt="Translate the following sentence to Urdu. Output only the translation, nothing else.\nSentence: 'The meeting is at 3pm tomorrow'",
    practice_num=5,
    practice_name="Specify output language and suppress explanation"
)

compare(
    bad_prompt="Extract info from: 'Ali, 25, Lahore, Engineer'",
    good_prompt="Extract the following fields from the text and return as JSON:\n- name\n- age\n- city\n- profession\n\nText: 'Ali, 25, Lahore, Engineer'",
    practice_num=6,
    practice_name="Name the exact fields you want extracted"
)

compare(
    bad_prompt="Fix my code:\ndef add(a,b)\n return a+b",
    good_prompt="Fix the syntax error in this Python code and return only the corrected code, no explanation:\ndef add(a,b)\n return a+b",
    practice_num=7,
    practice_name="Ask for code only when you don't need explanation"
)

compare(
    bad_prompt="Don't give me a long answer about climate change.",
    good_prompt="Explain climate change in exactly 2 sentences.",
    practice_num=8,
    practice_name="Use positive instructions, not negative ones"
)

compare(
    bad_prompt="Give me ideas.",
    good_prompt="Give me 5 startup ideas in the AI and education space for Pakistani university students. Each idea in one sentence.",
    practice_num=9,
    practice_name="Add domain and audience context"
)

compare(
    bad_prompt="Analyze this feedback and do something useful with it: 'App crashes on login'",
    good_prompt="""Analyze this user feedback and return a JSON with these fields:
- issue: one sentence description
- severity: Low/Medium/High/Critical  
- department: Technical/Billing/Logistics/UX
- suggested_action: one sentence

Feedback: 'App crashes on login'""",
    practice_num=10,
    practice_name="Structure output as JSON for programmatic use"
)