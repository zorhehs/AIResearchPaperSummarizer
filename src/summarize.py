import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import requests
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "llama3.2:1b"

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"
GROQ_THIRD_MODEL = "qwen/qwen3.8-27b"
# Each model has its OWN daily token budget on Groq's free tier. When one model
# is exhausted (daily quota), the next one still works — so instead of failing
# after one model's 429s, we rotate through the list below.
GROQ_MODELS = [GROQ_MODEL, GROQ_FALLBACK_MODEL, GROQ_THIRD_MODEL]

# gpt-oss-120b handles far more than this; beyond it we map-reduce instead
SINGLE_PASS_CHAR_LIMIT = 60000

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")


# Bump when summary shape/prompting changes so stale cached entries
# are not served forever.
# v6: key findings carry verbatim quotes for citation grounding.
CACHE_VERSION = "v6"

# ---------------------------------------------------------------------------
# Groq free-tier token budget
# ---------------------------------------------------------------------------
# This account's `openai/gpt-oss-120b` is limited to TPM_LIMIT tokens per
# minute. Sending a full paper (~35k+ chars ≈ 9k tokens) to the 6 parallel
# section writers blows that budget almost instantly, so every request returns
# 413 "Request too large" / 429 "Rate limit reached", burns ~60s in retries,
# and only then surfaces an error. Two safeguards fix it:
#   1) MAX_INPUT_CHARS caps any single request so it can never exceed the
#      per-minute limit (a request larger than TPM_LIMIT fails outright).
#   2) _reserve_budget() pre-books each call's token cost against a sliding
#      one-minute window and paces callers, so parallel section/chunk calls can
#      no longer pile up and hard-fail.
TPM_LIMIT = 8000
CHARS_PER_TOKEN = 3.8
OUTPUT_TOKEN_EST = 500
MAX_INPUT_CHARS = 20000             # hard cap for any single request (≈5.2k tokens)
SECTION_CONTEXT_CHARS = 18000       # excerpt handed to the section generator

_budget_lock = threading.Lock()
_budget = {"minute": int(time.time() // 60), "tokens": 0}


def _est_tokens(chars: int) -> int:
    return max(1, int(chars / CHARS_PER_TOKEN))


def _reserve_budget(tokens: int):
    """Block until `tokens` can be spent in the current one-minute window."""
    while True:
        with _budget_lock:
            now = int(time.time() // 60)
            if _budget["minute"] != now:
                _budget["minute"] = now
                _budget["tokens"] = 0
            if _budget["tokens"] + tokens <= TPM_LIMIT:
                _budget["tokens"] += tokens
                return
        time.sleep(1.0)


def _cache_get(text: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_cache (
                text_hash TEXT PRIMARY KEY,
                result_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        row = conn.execute(
            "SELECT result_json FROM summary_cache WHERE text_hash = ?",
            (hashlib.sha256(f"{CACHE_VERSION}:{text}".encode()).hexdigest(),),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _cache_put(text: str, result: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_cache (
                text_hash TEXT PRIMARY KEY,
                result_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        payload = {k: v for k, v in result.items() if k != "full_text"}
        conn.execute(
            "INSERT OR REPLACE INTO summary_cache (text_hash, result_json) VALUES (?, ?)",
            (hashlib.sha256(f"{CACHE_VERSION}:{text}".encode()).hexdigest(), json.dumps(payload)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

TEMPLATE_PHRASES = [
    "a 150-word paragraph summary",
    "a 100-word paragraph explanation",
    "an 80-word paragraph explanation",
    "finding 1",
]

META_PHRASES = [
    "i can provide",
    "i can analyze",
    "i'll provide",
    "here is a step-by-step",
    "certainly!",
    "sure, here",
]


RATE_LIMIT_MAX_RETRIES = 4


def _rate_limit_wait(error_msg: str) -> float:
    """Extract the suggested wait time (seconds) from a Groq 429 message, if present."""
    # Handles "try again in 19m39.792s" (daily) and "try again in 20.0s" (minute)
    m = re.search(r"try again in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", error_msg)
    if m:
        minutes = int(m.group(1) or 0)
        return minutes * 60 + float(m.group(2)) + 1.0
    return 10.0


def _ask_groq(messages: list, model: str = None, max_retries: int = 3) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not set in .env")
    client = Groq(api_key=api_key)

    # Cap combined input so a single request never exceeds the per-minute token
    # limit (otherwise Groq returns 413 "Request too large" and the request is
    # impossible no matter how long we wait).
    total = 0
    capped = []
    for m in messages:
        c = m.get("content") or ""
        if total + len(c) > MAX_INPUT_CHARS:
            c = c[: max(0, MAX_INPUT_CHARS - total)]
        total += len(c)
        capped.append({**m, "content": c})
    messages = capped

    # Reserve this request's token cost against the one-minute window up front.
    _reserve_budget(_est_tokens(total) + OUTPUT_TOKEN_EST)

    # Candidate models: the requested one first (if any), then the rotation list.
    candidates = []
    if model:
        candidates.append(model)
    candidates += [m for m in GROQ_MODELS if m not in candidates]

    last_error = "unknown Groq error"
    for candidate in candidates:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=messages,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                if "model_not_found" in error_str or "does not exist" in error_str:
                    break  # model invalid → try next
                if "rate_limit" in error_str.lower() or "429" in error_str or "413" in error_str:
                    # Daily quota exhausted for this model → move on, don't wait minutes.
                    if "tokens per day" in error_str.lower():
                        break
                    wait = _rate_limit_wait(error_str)
                    if wait > 20:  # multi-minute window on a busy path → try next model
                        break
                    if attempt == max_retries:
                        break
                    print(f"  (rate limited, waiting {wait:.1f}s... {candidate} attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                    continue
                break  # other API error → try next model
    raise Exception(f"Groq error: {last_error}")


def _is_template_echo(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in TEMPLATE_PHRASES)


def _is_meta_response(value: str) -> bool:
    """Detect responses that talk about answering instead of answering."""
    head = value.strip()[:120].lower()
    return any(phrase in head for phrase in META_PHRASES)


def _ask_ollama(prompt: str, json_mode: bool = True) -> str:
    payload = {
        "model": LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise Exception("Ollama is not running. Please start the Ollama app on your Mac.")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")

SYSTEM_PROMPT = """You are an expert AI research assistant. You analyze research papers and produce rigorous, academic summaries.
Follow the user's format instructions exactly. Never copy the instructions or example placeholders into your answer.
Always write real content grounded in the paper text provided. If the paper does not cover something, say so briefly rather than inventing it."""

# ---------------------------------------------------------------------------
# Structured paper summary — schema-validated JSON output
# ---------------------------------------------------------------------------

class KeyFinding(BaseModel):
    finding: str
    detail: Optional[str] = None
    quote: Optional[str] = None


class ResultRow(BaseModel):
    metric: str
    value: str
    comparison: Optional[str] = None


class KeyTerm(BaseModel):
    term: str
    definition: str


class PaperSummary(BaseModel):
    title: str
    authors: List[str] = []
    institutions: List[str] = []
    publication_info: Optional[str] = None
    one_line_summary: str
    field_tags: List[str] = []
    overview: str
    problem_statement: str
    approach: str
    key_findings: List[KeyFinding] = []
    results_table: List[ResultRow] = []
    significance: str
    limitations: List[str] = []
    future_work: List[str] = []
    key_terms: List[KeyTerm] = []
    confidence_notes: Optional[str] = None


PAPER_SUMMARY_SYSTEM_PROMPT = """You are a research paper summarization engine for an academic tool. You will be given the full text of a research paper (possibly including OCR artifacts from PDF extraction). Your job is to produce a structured, accurate summary.

RULES:
- Base every claim strictly on the provided text. Never invent authors, numbers, or findings.
- Paraphrase in your own words — do not copy sentences verbatim from the paper.
- Keep technical terms accurate; do not oversimplify to the point of being wrong.
- If a field cannot be supported by the text (e.g. limitations, institutions), return an empty value — do not fabricate.
- Numbers, equations, and results should be reported precisely as given.
- For every key finding, include "quote": a short verbatim quote (max 25 words) copied EXACTLY, word-for-word, from the paper text that supports the finding. The tool verifies these quotes against the text and flags any that cannot be found, so never paraphrase, trim, or invent a quote — if you cannot locate one, leave it empty.
- Write for an audience that is technically literate but not necessarily expert in this exact subfield — define jargon briefly in the key_terms field.
- Output ONLY valid JSON matching this schema — no markdown, no preamble, no code fences, no trailing commentary:
{"title": str, "authors": [str], "institutions": [str], "publication_info": str (journal/arXiv id/date if available, else ""), "one_line_summary": str (max 25 words), "field_tags": [str], "overview": str (2-4 sentences, what the paper is about and why it matters), "problem_statement": str (the gap or question motivating the work), "approach": str (2-4 sentences on methodology/technique), "key_findings": [{"finding": str, "detail": str (optional supporting number/context), "quote": str (verbatim, max 25 words, else "")}], "results_table": [{"metric": str, "value": str, "comparison": str (optional)}], "significance": str (why this matters / who should care), "limitations": [str], "future_work": [str], "key_terms": [{"term": str, "definition": str (one sentence)}], "confidence_notes": str (optional — flag ambiguous, truncated, or hard-to-parse parts; empty string if none)}"""


def _parse_json_object(raw: str) -> dict:
    """Extract a JSON object from a model response, tolerating code fences."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def _generate_paper_summary(text: str, title: str = "", abstract: str = "", source: str = "") -> dict:
    """Produce the full structured PaperSummary in ONE consolidated call.

    The model output is validated against the pydantic schema; invalid or
    unparseable JSON gets one corrective retry, then a local Ollama fallback.
    """
    context = _prepare_section_context(text)
    header = "Summarize the following research paper.\n"
    if title:
        header += f"\nTITLE (if known): {title}\n"
    if source:
        header += f"SOURCE: {source}\n"
    if abstract:
        header += f"ABSTRACT: {abstract[:800]}\n"
    user_content = header + "\nFULL TEXT:\n" + context
    messages = [
        {"role": "system", "content": PAPER_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if len(text) > SECTION_CONTEXT_CHARS:
        messages.append({"role": "user", "content":
            "NOTE: the paper text above was truncated to fit the context window "
            "(middle portions omitted). Fill whatever fields the available text "
            "supports and mention the truncation in confidence_notes."})

    last_err = "no response"
    for attempt in range(2):
        try:
            raw = _ask_groq(messages)
        except Exception as e:
            # provider-level failure (quota, network) → try the local fallback
            last_err = str(e)
            break
        data = _parse_json_object(raw)
        try:
            return PaperSummary(**data).model_dump()
        except ValidationError as e:
            last_err = f"schema validation failed ({e.error_count()} error(s))"
        except Exception as e:
            last_err = str(e)
        if attempt == 0:
            messages.append({"role": "assistant", "content": (raw or "")[:4000]})
            messages.append({"role": "user", "content":
                "Your last output was invalid JSON or did not match the schema. "
                "Return ONLY the JSON object matching the schema — no markdown, "
                "no preamble, no code fences."})

    # Last resort: local Ollama in JSON mode
    try:
        fallback = _ask_ollama(PAPER_SUMMARY_SYSTEM_PROMPT + "\n\n" + user_content, json_mode=True)
        return PaperSummary(**_parse_json_object(fallback)).model_dump()
    except Exception:
        pass
    raise Exception(f"Summary generation failed: {last_err}")


def _prepare_section_context(text: str) -> str:
    """Return a bounded excerpt that still covers the whole paper.

    Keeps the single generation request inside the account's per-minute/token
    budget. Short papers pass through unchanged; very long ones are sampled
    from head (abstract/intro), middle (methodology), and tail (results/
    conclusion) so the summary reflects the whole paper without the many extra
    API calls map-reduce would cost.
    """
    if len(text) <= SECTION_CONTEXT_CHARS:
        return text[:MAX_INPUT_CHARS]
    third = SECTION_CONTEXT_CHARS // 3
    head = text[:third]
    mid_start = (len(text) - third) // 2
    mid = text[mid_start:mid_start + third]
    tail = text[-third:]
    return "\n\n[ ... middle section omitted ... ]\n\n".join((head, mid, tail))[:MAX_INPUT_CHARS]


def summarize_paper(text: str, session_id: str = None, title: str = "", abstract: str = "", source: str = "") -> dict:
    if session_id:
        from user_session import check_and_increment_usage
        check_and_increment_usage(session_id)

    # Same paper content → reuse the cached summary
    cached = _cache_get(text)
    if cached:
        result = dict(cached)
        result["full_text"] = text
        result["cached"] = True
        return result

    result = _generate_paper_summary(text, title=title, abstract=abstract, source=source)
    result["full_text"] = text
    _cache_put(text, result)
    return result


def answer_question(paper_text: str, question: str, chat_history: list = None) -> str:
    history_text = ""
    for msg in (chat_history or [])[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_text += f"{role}: {msg.get('content', '')}\n"

    history_block = ""
    if history_text:
        history_block = "Conversation so far:\n" + history_text + "\n"

    condensed_text = paper_text[:15000]
    prompt = f"""You are an expert AI research assistant answering questions about a research paper.
Answer based ONLY on the paper text below. If the answer is not in the paper, say so honestly.
Keep answers concise (under 200 words) and use plain text.

{history_block}Paper text:
{condensed_text}

Question: {question}

Answer:"""
    try:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            return _ask_groq(messages).strip()
        except Exception:
            return _ask_ollama(prompt, json_mode=False).strip()
    except Exception as e:
        raise Exception(f"Local LLM Error - {str(e)}")


def stream_summarize_paper(text: str, title: str = "", abstract: str = "", source: str = ""):
    """Streaming variant of summarize_paper.

    The summary is generated in one consolidated model call, so the stream
    yields real checkpoints the frontend can sync to: a `stage: generating`
    event right before the model call (with the model name and whether the
    context was truncated), then a single `done` event carrying the complete
    structured payload (or `error` if generation fails, e.g. every model's
    daily quota is exhausted). A cache hit skips straight to `done`.
    """
    cached = _cache_get(text)
    if cached:
        result = dict(cached)
        result["full_text"] = text
        result["cached"] = True
        yield {"type": "done", "result": result}
        return

    yield {
        "type": "stage",
        "stage": "generating",
        "model": GROQ_MODEL,
        "truncated": len(text) > SECTION_CONTEXT_CHARS,
    }

    try:
        result = _generate_paper_summary(text, title=title, abstract=abstract, source=source)
    except Exception as e:
        yield {"type": "error", "detail": str(e)}
        return

    result["full_text"] = text
    _cache_put(text, result)
    yield {"type": "done", "result": result}
