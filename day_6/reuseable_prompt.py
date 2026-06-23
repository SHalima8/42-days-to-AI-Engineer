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
# DELIVERABLE 5 — Reusable Prompt Template Library
# 5 templates: summarization, entity extraction, sentiment
#              analysis, code generation, data transformation
# ============================================================

# --- TEMPLATE 1: SUMMARIZATION ---
def summarize(text, max_bullets=3):
    prompt = f"""
ROLE: You are a concise document summarizer.
TASK: Summarize the following text into exactly {max_bullets} bullet points.
Each bullet point must be under 20 words.
OUTPUT FORMAT:
- [bullet 1]
- [bullet 2]
- [bullet 3]

TEXT:
{text}
"""
    return call_llm(prompt, "TEMPLATE 1 - Summarization")


# --- TEMPLATE 2: ENTITY EXTRACTION ---
def extract_entities(text):
    prompt = f"""
ROLE: You are a named entity recognition expert.
TASK: Extract all named entities from the text below.
OUTPUT FORMAT (strict):
PERSON: [names]
ORGANIZATION: [organizations]
LOCATION: [places]
DATE: [dates or time references]
MONEY: [monetary values]
If a category has no entities write NONE.

TEXT:
{text}
"""
    return call_llm(prompt, "TEMPLATE 2 - Entity Extraction")


# --- TEMPLATE 3: SENTIMENT ANALYSIS ---
def analyze_sentiment(text):
    prompt = f"""
ROLE: You are a sentiment analysis engine.
TASK: Analyze the sentiment of the text below.
OUTPUT FORMAT (strict):
Sentiment: [Positive/Negative/Neutral/Mixed]
Confidence: [High/Medium/Low]
Key Phrases: [2-3 phrases that drove this classification]
Reason: [One sentence explanation]

TEXT:
{text}
"""
    return call_llm(prompt, "TEMPLATE 3 - Sentiment Analysis")


# --- TEMPLATE 4: CODE GENERATION ---
def generate_code(task_description, language="Python"):
    prompt = f"""
ROLE: You are a senior {language} developer.
TASK: Write clean, well-commented {language} code for the following:
{task_description}
OUTPUT FORMAT:
- Code only, no explanation outside the code
- Add inline comments explaining each step
- Include a usage example at the bottom
"""
    return call_llm(prompt, "TEMPLATE 4 - Code Generation")


# --- TEMPLATE 5: DATA TRANSFORMATION ---
def transform_data(raw_data, output_format):
    prompt = f"""
ROLE: You are a data transformation specialist.
TASK: Convert the raw data below into {output_format} format.
Rules:
- Preserve all original values exactly
- Follow the output format strictly
- If a field is missing write null

RAW DATA:
{raw_data}

OUTPUT FORMAT REQUIRED: {output_format}
"""
    return call_llm(prompt, "TEMPLATE 5 - Data Transformation")


# ============================================================
# RUN ALL TEMPLATES WITH SAMPLE INPUTS
# ============================================================

# Template 1 — Summarization
summarize("""
Artificial intelligence is transforming industries worldwide. 
In healthcare, AI models detect diseases from medical scans with accuracy 
surpassing human doctors. In finance, algorithms trade stocks and detect fraud 
in milliseconds. In education, personalized AI tutors adapt to each student's 
learning pace. However, concerns about job displacement, data privacy, and 
algorithmic bias remain significant challenges that governments and companies 
must address together.
""", max_bullets=3)

# Template 2 — Entity Extraction
extract_entities(
    "Ahmed Khan from Islamabad joined Microsoft in January 2024 and received a signing bonus of $10,000."
)

# Template 3 — Sentiment Analysis
analyze_sentiment(
    "The internship has been intense but incredibly rewarding. Some tasks were confusing at first but the learning curve is worth it."
)

# Template 4 — Code Generation
generate_code(
    "A function that takes a list of numbers and returns the mean, median, and mode",
    language="Python"
)

# Template 5 — Data Transformation
transform_data(
    raw_data="""
    Name: Ali Raza
    Age: 24
    City: Lahore
    Job: AI Engineer
    Salary: 85000
    """,
    output_format="JSON"
)