import os
import json
from google import genai
from dotenv import load_dotenv

from tools_schemas import (
    calculate_declaration,
    search_database_declaration,
)
from tools_function import calculate, search_database

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

all_tools = [calculate_declaration, search_database_declaration]

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "search_database": search_database,
}

prompt = "Look up Sultana in the database, then tell me what 3 times her id number is."

interaction = client.interactions.create(
    model="gemini-3-flash-preview",
    input=prompt,
    tools=all_tools,
)

# Loop until the model stops asking for tools
while True:
    fc_steps = [s for s in interaction.steps if s.type == "function_call"]

    if not fc_steps:
        break  # model is done calling tools, ready to answer

    function_results = []
    for fc_step in fc_steps:
        print("Model requested:", fc_step.name, fc_step.arguments)
        func = TOOL_FUNCTIONS[fc_step.name]
        result = func(**fc_step.arguments)
        print("Result:", result)

        function_results.append({
            "type": "function_result",
            "name": fc_step.name,
            "call_id": fc_step.id,
            "result": [{"type": "text", "text": json.dumps(result)}],
        })

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=function_results,
        tools=all_tools,
        previous_interaction_id=interaction.id,
    )

print("Final answer:", interaction.output_text)