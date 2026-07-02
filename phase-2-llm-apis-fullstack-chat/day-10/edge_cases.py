import os
import json
from google import genai
from dotenv import load_dotenv

from tools_schemas import get_current_time_declaration, calculate_declaration
from tools_function import get_current_time, calculate

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Case 1: model declines to call any function ──────────────
# A prompt with no reason to use a tool at all
print("=== Case 1: model declines ===")
interaction = client.interactions.create(
    model="gemini-3-flash-preview",
    input="What's your favorite color?",
    tools=[get_current_time_declaration, calculate_declaration],
)
fc_steps = [s for s in interaction.steps if s.type == "function_call"]
if not fc_steps:
    print("No function call made. Model answered directly:")
    print(interaction.output_text)
else:
    print("Model called a tool unexpectedly:", fc_steps[0].name)


# ── Case 2: invalid argument type / bad input ─────────────────
# Force a bad timezone straight into your function, bypassing the model
print("\n=== Case 2: invalid argument ===")
bad_result = get_current_time(timezone="Not_A_Real_Timezone")
print(bad_result)


# ── Case 3: function raises an error you must catch ───────────
# Force a bad expression straight into calculate()
print("\n=== Case 3: function errors on bad input ===")
bad_calc = calculate(expression="3 * * 2")  # malformed on purpose
print(bad_calc)

bad_calc_chars = calculate(expression="import os; os.system('ls')")  # blocked chars
print(bad_calc_chars)