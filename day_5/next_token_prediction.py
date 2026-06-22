import torch
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

sentence = "The cat sat on the"

tokens = tokenizer(sentence)

print("Original Sentence:\n")
print(sentence)

print("\nToken IDs:\n")
print(tokens["input_ids"])

print("\nDecoded Tokens:\n")
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))

inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

last_logits = logits[:, -1, :]

probs = torch.softmax(last_logits, dim=-1)

top_probs, top_indices = torch.topk(probs, 5)


print("Current Context:\n")
print(sentence)

print("\nTop 5 Predictions\n")

for i in range(5):
    token = tokenizer.decode(top_indices[0][i])

    print(
        f"{i+1}. {repr(token)} : {top_probs[0][i].item():.4f}"
    )



    labels = [
    tokenizer.decode(idx)
    for idx in top_indices[0]
]

values = top_probs[0].tolist()

plt.figure(figsize=(8,4))
plt.bar(labels, values)

plt.title("Top-5 Next Token Probabilities")

plt.xlabel("Predicted Token")
plt.ylabel("Probability")

plt.show()