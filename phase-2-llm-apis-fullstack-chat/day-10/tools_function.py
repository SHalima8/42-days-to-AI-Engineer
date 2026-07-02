from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time(timezone: str) -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
        return {
            "timezone": timezone,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"error": f"Invalid timezone '{timezone}': {str(e)}"}
    

# add to tool_functions.py
def calculate(expression: str) -> dict:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return {"error": f"Expression contains disallowed characters: '{expression}'"}
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}
    
# add to tool_functions.py
CURRENCY_SYMBOLS = {"USD": "$", "GBP": "£", "PKR": "Rs", "EUR": "€"}

def format_currency(amount: float, currency_code: str) -> dict:
    symbol = CURRENCY_SYMBOLS.get(currency_code)
    if not symbol:
        return {"error": f"Unsupported currency code: '{currency_code}'"}
    return {"formatted": f"{symbol}{amount:,.2f}", "currency_code": currency_code}

# add to tool_functions.py
MOCK_DB = {
    "sultana": {"id": 1, "role": "AI Engineer Intern", "team": "NLP/TTS"},
    "planetbeyond": {"id": 2, "role": "Company", "team": "N/A"},
}

def search_database(query: str) -> dict:
    key = query.strip().lower()
    if key in MOCK_DB:
        return {"query": query, "found": True, "record": MOCK_DB[key]}
    return {"query": query, "found": False, "record": None}