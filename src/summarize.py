import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def _ask_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def summarize_text(text: str) -> str:
    prompt = f"""Summarize the following research paper in about 150 words.
Focus on what the paper is about, its core contribution, and why it matters.

Paper text:
{text}
"""
    return _ask_groq(prompt)


def extract_methodology(text: str) -> str:
    prompt = f"""Extract and describe the methodology used in this research paper.
Explain the approach, techniques, models, or experimental design used — in 100-150 words.

Paper text:
{text}
"""
    return _ask_groq(prompt)


def extract_research_gaps(text: str) -> str:
    prompt = f"""Identify the research gaps or limitations mentioned or implied in this paper.
What open questions remain, or what does the paper acknowledge it doesn't fully solve? Answer in 80-120 words.

Paper text:
{text}
"""
    return _ask_groq(prompt)


def extract_findings(text: str) -> str:
    prompt = f"""List the key findings or results of this research paper as 3-5 concise bullet points.

Paper text:
{text}
"""
    return _ask_groq(prompt)


def extract_future_work(text: str) -> str:
    prompt = f"""Extract or infer the future work directions suggested by this paper.
What do the authors suggest could be explored next? Answer in 80-120 words.

Paper text:
{text}
"""
    return _ask_groq(prompt)


def _get_relevant_text(sections: dict, keys: list, fallback: str) -> str:
    """Pull whichever of the given section keys exist and concatenate them.
    Falls back to the full condensed text if none of those keys were detected
    (e.g. when sections.py fell back to generic chunk_N splitting)."""
    parts = [sections[k] for k in keys if k in sections]
    combined = "\n\n".join(parts).strip()
    return combined if combined else fallback


def summarize_paper(text: str, session_id: str = None) -> dict:
    if session_id:
        from user_session import check_and_increment_usage
        check_and_increment_usage(session_id)

    from map_reduce import get_condensed_text
    condensed = get_condensed_text(text)

    from sections import get_sections
    sections = get_sections(text)

    # If sections.py fell back to generic chunk_N splitting, we have no
    # meaningful section labels -- just use the condensed full text everywhere.
    has_real_sections = not all(k.startswith("chunk_") for k in sections.keys())

    if has_real_sections:
        methodology_text = _get_relevant_text(sections, ["methodology", "method", "methods", "approach"], condensed)
        results_text = _get_relevant_text(sections, ["results", "evaluation", "main results", "experiments"], condensed)
        gaps_text = _get_relevant_text(sections, ["limitations", "discussion"], condensed)
        future_text = _get_relevant_text(sections, ["future work", "conclusion", "conclusions"], condensed)
    else:
        methodology_text = results_text = gaps_text = future_text = condensed

    return {
        "summary": summarize_text(condensed),
        "methodology": extract_methodology(methodology_text),
        "research_gaps": extract_research_gaps(gaps_text),
        "findings": extract_findings(results_text),
        "future_work": extract_future_work(future_text),
    }


if __name__ == "__main__":
    from mock_sections import get_mock_sections

    sections = get_mock_sections()
    result = summarize_paper(sections["full_text"])

    for key, value in result.items():
        print(f"\n===== {key.upper()} =====")
        print(value)
