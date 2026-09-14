# Groq vs Ollama Benchmark

Comparing cloud (Groq, `openai/gpt-oss-20b`) vs local (Ollama, `llama3.2:1b`) summarization on 5 real papers.

| Paper | Chars | Groq time | Groq status | Ollama time | Ollama status |
|---|---|---|---|---|---|
| paper1.pdf | 55297 | 64.5s | FAILED: Summary generation failed: schema valida | 0.2s | FAILED: 6 validation errors for PaperSummary
tit |
| paper2.pdf | 34998 | 113.4s | FAILED: Summary generation failed: schema valida | 2.7s | FAILED: 5 validation errors for PaperSummary
one |
| paper3.pdf | 64546 | 93.7s | ok | 2.2s | FAILED: 5 validation errors for PaperSummary
one |
| paper4.pdf | 84603 | 57.8s | ok | 1.9s | FAILED: 5 validation errors for PaperSummary
one |
| paper5.pdf | 51059 | 117.3s | ok | 2.1s | FAILED: 6 validation errors for PaperSummary
tit |

## Analysis

Ollama (`llama3.2:1b`) failed schema validation on all 5 papers, and its
response times (0.2s–2.7s) are too fast to represent genuine full-paper
analysis — this strongly suggests the model is returning early,
incomplete JSON rather than actually working through the paper text and
the full `PaperSummary` schema (14 fields, several nested).

**Conclusion:** the current Ollama fallback is well-suited to simpler,
single-field tasks (e.g. the free-text `/chat` answers in
`answer_question()`), but `llama3.2:1b` is too small to reliably produce
the full structured multi-field paper summary. Options worth exploring:
- A larger local model (e.g. `llama3.1:8b` or `qwen2.5:7b`) at the cost
  of slower inference
- A simplified summary schema specifically for the Ollama fallback path
  (fewer required fields)
- Splitting the Ollama fallback into multiple smaller, sequential calls
  instead of one large structured-JSON request
