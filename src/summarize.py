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
{text[:8000]}
"""
    return _ask_groq(prompt)


def extract_methodology(text: str) -> str:
    prompt = f"""Extract and describe the methodology used in this research paper.
Explain the approach, techniques, models, or experimental design used — in 100-150 words.

Paper text:
{text[:8000]}
"""
    return _ask_groq(prompt)


def extract_research_gaps(text: str) -> str:
    prompt = f"""Identify the research gaps or limitations mentioned or implied in this paper.
What open questions remain, or what does the paper acknowledge it doesn't fully solve? Answer in 80-120 words.

Paper text:
{text[:8000]}
"""
    return _ask_groq(prompt)


def extract_findings(text: str) -> str:
    prompt = f"""List the key findings or results of this research paper as 3-5 concise bullet points.

Paper text:
{text[:8000]}
"""
    return _ask_groq(prompt)


def extract_future_work(text: str) -> str:
    prompt = f"""Extract or infer the future work directions suggested by this paper.
What do the authors suggest could be explored next? Answer in 80-120 words.

Paper text:
{text[:8000]}
"""
    return _ask_groq(prompt)


def summarize_paper(text: str) -> dict:
    from map_reduce import get_condensed_text
    text = get_condensed_text(text)
    return {
        "summary": summarize_text(text),
        "methodology": extract_methodology(text),
        "research_gaps": extract_research_gaps(text),
        "findings": extract_findings(text),
        "future_work": extract_future_work(text),
    }


if __name__ == "__main__":
    from mock_sections import get_mock_sections

    sections = get_mock_sections()
    result = summarize_paper(sections["full_text"])

    for key, value in result.items():
        print(f"\n===== {key.upper()} =====")
        print(value)
