# Cross-Model Comparison: GPT vs Claude vs Gemini

Comparing one mid-tier, cost-efficient model per provider — the kind realistically used in a production STT/TTS or chat product, not the most expensive flagship.

| Model | Provider | Context Window | Max Output Tokens | Input Price (per 1M tokens) | Output Price (per 1M tokens) |
|---|---|---|---|---|---|
| GPT-4o-mini | OpenAI | 128K tokens | 16,384 tokens | $0.15 | $0.60 |
| Claude Sonnet 4.6 | Anthropic | 1,000,000 tokens | 64,000 tokens | $3.00 | $15.00 |
| Gemini 2.5 Flash-Lite | Google | (check AI Studio for exact figure at time of testing) | — | $0.10 | $0.40 |

**Observations:**

- Gemini 2.5 Flash-Lite is the cheapest of the three on a per-token basis, and is the model I used throughout this task.
- GPT-4o-mini has the smallest context window by far (128K vs Claude's 1M) — this matters for tasks like summarizing very long transcripts, where Claude's window could hold an entire long audio transcript in one call without chunking.
- Claude Sonnet 4.6 is meaningfully more expensive per token than the other two, but its 1M context window comes at standard pricing with no long-context surcharge — competitors often charge more once you exceed certain context thresholds.
- For a low-resource-language STT/TTS pipeline doing high-volume, simple tasks (e.g., transcript cleanup, short summaries), Gemini Flash-Lite's price point is hard to beat. For tasks needing to process very long context in one shot (e.g., a full meeting transcript), Claude's 1M window is the standout feature, independent of price.