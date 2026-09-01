import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import requests
from dotenv import load_dotenv

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
# (e.g. pre-key_stats 5-section summaries) are not served forever.
CACHE_VERSION = "v3"

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

PAPER_TEXT_RULE = "Base your answer ONLY on the paper text below."

# Each of the 5 sections uses its own dedicated prompting + formatting strategy.
SECTION_STRATEGIES = {
    "summary": {
        "instruction": """Write the summary as a single cohesive academic paragraph of about 150 words, in the style of a journal abstract.
Cover, in this order: the problem the paper addresses, the approach it takes, and the significance of its results.
Plain prose only — no bullet points, no headings.""",
        "format": "paragraph",
    },
    "methodology": {
        "instruction": """Describe the methodology as a numbered list of 3-6 steps (1., 2., 3., ...), one step per line.
Each step is one sentence describing a distinct methodological component: data, model/architecture, experimental design, or evaluation metrics.
Begin with one short introductory sentence before the list.""",
        "format": "numbered_steps",
    },
    "research_gaps": {
        "instruction": """Identify limitations and open questions as a bulleted list of 3-5 items (one per line, each starting with "- ").
Each bullet states one specific gap and, in a few words, why it matters.
Keep each bullet under 30 words.""",
        "format": "bullet_list",
    },
    "findings": {
        "instruction": """Report the key findings as a bulleted list of 3-5 items (one per line, each starting with "- ").
Each bullet is a single concrete, specific finding stated as a complete sentence with the quantitative result where available (e.g. "achieved 34.2 BLEU, +4.1 over the baseline").
Avoid vague statements like "the method works well".""",
        "format": "bullet_list",
    },
    "future_work": {
        "instruction": """Write the future-work outlook as one short paragraph of 2-3 sentences, followed by 2-4 concrete suggested directions as a bulleted list (one per line, each starting with "- ").
Each suggested direction should be actionable, not generic.""",
        "format": "paragraph_plus_list",
    },
    "key_stats": {
        "instruction": """Extract the key quantitative results as a bulleted list of 3-6 items (one per line, each starting with "- ").
Each bullet follows the pattern "metric: value (context/comparison)" — e.g. "Accuracy: 94.2% (+3.1 over the BERT baseline on GLUE)".
Include dataset sizes, sample counts, or runtime figures where the paper reports them.
If the paper reports no quantitative results, write exactly: "No quantitative results reported."
Copy numbers verbatim from the paper — never round, convert, or estimate.""",
        "format": "bullet_list",
    },
}

_SECTION_ORDER = ["summary", "methodology", "research_gaps", "findings", "future_work", "key_stats"]


def _paper_context_block(title: str = "", abstract: str = "") -> str:
    """Optional identity block prepended to section prompts so the model can
    anchor its analysis to the paper instead of writing generic prose."""
    parts = []
    if title:
        parts.append(f"Paper title: {title}")
    if abstract:
        parts.append(f"Paper abstract: {abstract[:1500]}")
    return ("\n".join(parts) + "\n") if parts else ""


def _ask_section(section: str, condensed_text: str, title: str = "", abstract: str = "") -> str:
    strategy = SECTION_STRATEGIES[section]
    prompt = f"""{_paper_context_block(title, abstract)}
{strategy["instruction"]}

{PAPER_TEXT_RULE}

Paper text:
{condensed_text}
"""
    last_error = None
    for attempt in range(2):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]
            if attempt == 1:
                messages.append({"role": "user", "content":
                    "Your previous response copied placeholder text from the instructions instead of analyzing the paper. "
                    "Do NOT repeat the instructions. Write real analysis of the paper text only."})
            response_text = _ask_groq(messages).replace("```json", "").replace("```", "").strip()
            if response_text and not _is_template_echo(response_text):
                # A findings section without a single number is almost always
                # vague filler — retry once asking for the quantitative results.
                if section == "findings" and attempt == 0 and not any(ch.isdigit() for ch in response_text):
                    messages.append({"role": "user", "content":
                        "Your response contains no concrete numbers. Re-answer with the paper's actual "
                        "quantitative results (metrics, percentages, dataset sizes). If the paper truly "
                        "reports none, say so explicitly."})
                    retry_text = _ask_groq(messages).replace("```json", "").replace("```", "").strip()
                    if retry_text and any(ch.isdigit() for ch in retry_text):
                        response_text = retry_text
                # Strip a leading meta sentence like "I can provide a step-by-step analysis..."
                if _is_meta_response(response_text):
                    lines = response_text.split("\n")
                    if len(lines) > 1:
                        response_text = "\n".join(lines[1:]).strip()
                    else:
                        last_error = "meta response"
                        continue
                return response_text
            last_error = "template echo or empty response"
        except Exception as e:
            last_error = str(e)
            break

    # Fallback to local Ollama for this section
    try:
        fallback = _ask_ollama(prompt, json_mode=False).strip()
        if fallback:
            return fallback
    except Exception as e:
        last_error = f"{last_error} | Ollama: {str(e)}"
    raise Exception(f"Section '{section}' failed: {last_error}")


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


def _generate_all_sections(text: str, title: str = "", abstract: str = "") -> dict:
    """Ask the model for every section in a SINGLE request.

    The old flow (map-reduce every long paper, then fire 6 parallel section
    calls, each resending the full paper) burned ~25k+ tokens and 7+ calls per
    paper, which melted Groq's 8k/min and 200k/day budgets and caused the
    minutes-long failures. One consolidated JSON call is ~5k tokens and finishes
    in seconds. `_ask_groq` rotates across models when one is rate-limited.
    """
    strategy_lines = "\n".join(
        f"- {key}: {SECTION_STRATEGIES[key]['instruction'].strip()}"
        for key in _SECTION_ORDER
    )
    context = _paper_context_block(title, abstract)
    user_content = (
        "Return ONLY valid JSON — no prose, no markdown fences. "
        f"The JSON must have exactly these keys: {', '.join(_SECTION_ORDER)}.\n\n"
        "Content requirements per section:\n" + strategy_lines + "\n\n"
        + context
        + "\n\nPaper text:\n" + text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + " Always return valid JSON with the requested keys."},
        {"role": "user", "content": user_content},
    ]

    def _parse(raw: str) -> dict:
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

    raw = _ask_groq(messages)
    data = _parse(raw)

    # One corrective retry when sections are missing/empty
    missing = [k for k in _SECTION_ORDER if not data.get(k)]
    if missing:
        try:
            retry_content = (
                "Return ONLY valid JSON with exactly these keys: "
                f"{', '.join(_SECTION_ORDER)}. You omitted or left empty: {', '.join(missing)}.\n"
                "Base the content on this paper excerpt:\n" + text[:8000]
            )
            retry = _parse(_ask_groq([{"role": "user", "content": retry_content}]))
            for k in missing:
                if retry.get(k):
                    data[k] = retry[k]
        except Exception:
            pass

    # Last-resort: generate any still-missing sections individually (rare).
    for k in [k for k in _SECTION_ORDER if not data.get(k)]:
        try:
            data[k] = _ask_section(k, text[:6000], title=title, abstract=abstract)
        except Exception:
            data[k] = ""
    return data


def _as_text(v) -> str:
    """Coerce a model-provided JSON value (string, list, or dict) to text."""
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, dict):
                parts.append(x.get("content") or x.get("text") or x.get("value") or "")
            else:
                parts.append(str(x))
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict):
        return v.get("content") or v.get("text") or v.get("value") or ""
    return "" if v is None else str(v)


def _polish_sections(data: dict, text: str, title: str = "", abstract: str = "") -> dict:
    """Apply the quality guards per section (echo/meta/findings-digit) to a
    consolidated JSON result."""
    sections = {}
    for k in _SECTION_ORDER:
        v = _as_text(data.get(k)).strip()
        if not v or _is_template_echo(v):
            v = ""
        elif _is_meta_response(v):
            lines = v.split("\n")
            v = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        sections[k] = v

    # A findings section without a single number is almost always vague filler —
    # retarget once asking for the quantitative results.
    if sections.get("findings") and not any(ch.isdigit() for ch in sections["findings"]):
        try:
            retry = _ask_section("findings", text[:6000], title=title, abstract=abstract)
            if retry and any(ch.isdigit() for ch in retry):
                sections["findings"] = retry
        except Exception:
            pass
    return sections


def summarize_paper(text: str, session_id: str = None, title: str = "", abstract: str = "") -> dict:
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

    condensed_text = _prepare_section_context(text)
    data = _generate_all_sections(condensed_text, title=title, abstract=abstract)
    sections = _polish_sections(data, condensed_text, title=title, abstract=abstract)

    result = {k: (v if v else "Data not provided by the model.") for k, v in sections.items()}
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


def stream_summarize_paper(text: str, title: str = "", abstract: str = ""):
    """Streaming variant of summarize_paper.

    Now backed by a single consolidated model call that returns every section
    at once (see _generate_all_sections). Yields a `section_done` event for each
    section, then a final `done` event with the complete summary payload. Yields
    an `error` event instead if generation fails (e.g. every model's daily
    quota is exhausted).
    """
    cached = _cache_get(text)
    if cached:
        result = dict(cached)
        result["full_text"] = text
        result["cached"] = True
        yield {"type": "done", "result": result}
        return

    try:
        condensed_text = _prepare_section_context(text)
        data = _generate_all_sections(condensed_text, title=title, abstract=abstract)
        sections = _polish_sections(data, condensed_text, title=title, abstract=abstract)
    except Exception as e:
        yield {"type": "error", "detail": str(e)}
        return

    for k in _SECTION_ORDER:
        yield {"type": "section_done", "section": k, "content": sections.get(k, "")}

    result = {k: (v if v else "Data not provided by the model.") for k, v in sections.items()}
    result["full_text"] = text
    _cache_put(text, result)
    yield {"type": "done", "result": result}
