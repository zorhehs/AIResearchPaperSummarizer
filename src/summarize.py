import hashlib
import json
import os
import re
import sqlite3
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "llama3.2:1b"

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"

# gpt-oss-120b handles far more than this; beyond it we map-reduce instead
SINGLE_PASS_CHAR_LIMIT = 60000

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")


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
            (hashlib.sha256(text.encode()).hexdigest(),),
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
            (hashlib.sha256(text.encode()).hexdigest(), json.dumps(payload)),
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
    match = re.search(r"try again in ([\d.]+)s", error_msg)
    if match:
        return float(match.group(1)) + 1.0
    return 20.0


def _ask_groq(messages: list) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not set in .env")
    client = Groq(api_key=api_key)

    model = GROQ_MODEL
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "model_not_found" in error_str or "does not exist" in error_str:
                model = GROQ_FALLBACK_MODEL
                continue
            # Rate limit (429): wait the suggested time and retry on the same model
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise Exception(f"Groq error: {error_str}")
                wait = _rate_limit_wait(error_str)
                print(f"  (rate limited, waiting {wait:.1f}s... attempt {attempt}/{RATE_LIMIT_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise Exception(f"Groq error: {error_str}")


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
}

_SECTION_ORDER = ["summary", "methodology", "research_gaps", "findings", "future_work"]


def _ask_section(section: str, condensed_text: str) -> str:
    strategy = SECTION_STRATEGIES[section]
    prompt = f"""{strategy["instruction"]}

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


def summarize_paper(text: str, session_id: str = None) -> dict:
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

    # Short enough → one pass with the full text; longer → map-reduce first
    if len(text) > SINGLE_PASS_CHAR_LIMIT:
        import map_reduce
        condensed_text = map_reduce.get_condensed_text(text)
    else:
        condensed_text = text[:SINGLE_PASS_CHAR_LIMIT]

    # Generate each of the 5 sections with its own strategy, in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = dict(zip(
            _SECTION_ORDER,
            executor.map(lambda s: _ask_section(s, condensed_text), _SECTION_ORDER),
        ))

    result = {k: (v if v else "Data not provided by the model.") for k, v in results.items()}
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


def stream_summarize_paper(text: str):
    """Streaming variant of summarize_paper.

    Yields {"type": "section_done", "section": ..., "content": ...} as each of the
    5 sections finishes (they still run in parallel), then a final
    {"type": "done", "result": {...}} event with the complete summary payload.
    Yields {"type": "error", "detail": ...} instead if every section failed.
    Mirrors summarize_paper: cache first, map-reduce for long papers.
    """
    cached = _cache_get(text)
    if cached:
        result = dict(cached)
        result["full_text"] = text
        result["cached"] = True
        yield {"type": "done", "result": result}
        return

    if len(text) > SINGLE_PASS_CHAR_LIMIT:
        import map_reduce
        condensed_text = map_reduce.get_condensed_text(text)
    else:
        condensed_text = text[:SINGLE_PASS_CHAR_LIMIT]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_ask_section, s, condensed_text): s for s in _SECTION_ORDER}
        for fut in as_completed(futures):
            section = futures[fut]
            try:
                value = fut.result()
            except Exception as e:
                value = ""
                errors[section] = str(e)
            results[section] = value
            yield {"type": "section_done", "section": section, "content": value}

    # All sections failed → surface the error (mirrors summarize_paper raising)
    if errors and len(errors) == len(_SECTION_ORDER):
        raise Exception(f"Summarization failed: {errors.get('summary', next(iter(errors.values())))}")

    result = {k: (v if v else "Data not provided by the model.") for k, v in results.items()}
    result["full_text"] = text
    _cache_put(text, result)
    yield {"type": "done", "result": result}
