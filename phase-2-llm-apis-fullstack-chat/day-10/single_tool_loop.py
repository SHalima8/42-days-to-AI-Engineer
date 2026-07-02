import os
import json
from google import genai
from dotenv import load_dotenv

from tools_schemas import calculate_declaration
from tools_function import calculate

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

interaction = client.interactions.create(
    model="gemini-3-flash-preview",
    input="What is (15 + 3) times 2?",
    tools=[calculate_declaration],
)

fc_step = next(s for s in interaction.steps if s.type == "function_call")
print("Model requested:", fc_step.name, fc_step.arguments)

if fc_step.name == "calculate":
    result = calculate(**fc_step.arguments)
    print("Function result:", result)

final_interaction = client.interactions.create(
    model="gemini-3-flash-preview",
    input=[{
        "type": "function_result",
        "name": fc_step.name,
        "call_id": fc_step.id,
        "result": [{"type": "text", "text": json.dumps(result)}],
    }],
    tools=[calculate_declaration],
    previous_interaction_id=interaction.id,
)

print("Final answer:", final_interaction.output_text)