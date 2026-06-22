import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# -------------------------------------------------------
# Load GPT-2
# -------------------------------------------------------
print("Loading GPT-2...\n")

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

# -------------------------------------------------------
# Prompt
# -------------------------------------------------------

prompt = (
    "Once upon a time, a robot discovered an old piano in an abandoned city."
)

inputs = tokenizer(prompt, return_tensors="pt")


# -------------------------------------------------------
# Helper Function
# -------------------------------------------------------

def generate(title, **kwargs):

    print("=" * 80)
    print(title)
    print("=" * 80)

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print(text)
    print("\n\n")


# =======================================================
# Temperature
# =======================================================

generate(
    "Temperature = 0.2 (Very Deterministic)",
    temperature=0.2
)

generate(
    "Temperature = 0.7 (Balanced)",
    temperature=0.7
)

generate(
    "Temperature = 1.3 (Creative)",
    temperature=1.3
)

# =======================================================
# Top-k
# =======================================================

generate(
    "Top-k = 5",
    top_k=5,
    temperature=1
)

generate(
    "Top-k = 20",
    top_k=20,
    temperature=1
)

generate(
    "Top-k = 50",
    top_k=50,
    temperature=1
)

# =======================================================
# Top-p
# =======================================================

generate(
    "Top-p = 0.5",
    top_p=0.5,
    temperature=1
)

generate(
    "Top-p = 0.8",
    top_p=0.8,
    temperature=1
)

generate(
    "Top-p = 0.95",
    top_p=0.95,
    temperature=1
)