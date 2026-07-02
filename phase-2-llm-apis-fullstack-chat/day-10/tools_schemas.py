get_current_time_declaration = {
    "type": "function",
    "name": "get_current_time",
    "description": "Gets the current date and time for a given IANA timezone (e.g. 'Asia/Karachi', 'UTC', 'America/New_York'). Use this whenever the user asks what time or date it is somewhere.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Asia/Karachi', 'UTC', 'America/New_York'",
            },
        },
        "required": ["timezone"],
    },
}

# add to tool_schemas.py
calculate_declaration = {
    "type": "function",
    "name": "calculate",
    "description": "Evaluates a basic arithmetic expression (e.g. '12 * 4 + 7') and returns the numeric result. Use this whenever the user asks for a calculation instead of doing math yourself.",
    "parameters": { 
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A basic arithmetic expression using +, -, *, /, and parentheses, e.g. '(15 + 3) * 2'",
            },
        },
        "required": ["expression"],
    },
}

# add to tool_schemas.py
search_database_declaration = {
    "type": "function",
    "name": "search_database",
    "description": "Searches a mock customer/product database by a search term and returns matching records. Use this when the user asks to look up or find stored data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term to look up in the database, e.g. a name or product ID.",
            },
        },
        "required": ["query"],
    },
}

# add to tool_schemas.py
format_currency_declaration = {
    "type": "function",
    "name": "format_currency",
    "description": "Formats a numeric amount into a currency string for a given currency code. Use this when the user wants an amount displayed as money.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "The numeric amount to format, e.g. 1500.5",
            },
            "currency_code": {
                "type": "string",
                "enum": ["USD", "GBP", "PKR", "EUR"],
                "description": "The 3-letter currency code to format the amount in.",
            },
        },
        "required": ["amount", "currency_code"],
    },
}