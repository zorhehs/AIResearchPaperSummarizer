"""
Benchmark: Groq (cloud) vs Ollama (local) for paper summarization.

Runs the same set of papers through both providers, measuring wall-clock
time and basic output characteristics (length, whether it parsed as valid
JSON matching the PaperSummary schema). Results are written to
docs/benchmark.md so they're easy to share/reference.

Usage:
    python scripts/benchmark_groq_vs_ollama.py
"""
import os
import sys
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from extract import extract_text
from clean import clean_text
import summarize as summarize_module


def _time_call(fn, *args, **kwargs):
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        return {"success": True, "elapsed": elapsed, "result": result, "error": None}
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "elapsed": elapsed, "result": None, "error": str(e)}


def benchmark_paper(pdf_path):
    name = os.path.basename(pdf_path)
    print("\n=== " + name + " ===")

    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)
    print("  text length: " + str(len(cleaned)) + " chars")

    print("  running via Groq...")
    groq_run = _time_call(summarize_module._generate_paper_summary, cleaned)
    if groq_run["success"]:
        print("  Groq:   " + str(round(groq_run["elapsed"], 1)) + "s  (ok)")
    else:
        print("  Groq:   " + str(round(groq_run["elapsed"], 1)) + "s  (FAILED: " + str(groq_run["error"]) + ")")

    print("  running via Ollama (forced)...")
    context = summarize_module._prepare_section_context(cleaned)
    prompt = summarize_module.PAPER_SUMMARY_SYSTEM_PROMPT + "\n\nFULL TEXT:\n" + context

    def _ollama_call():
        raw_out = summarize_module._ask_ollama(prompt, json_mode=True)
        data = summarize_module._parse_json_object(raw_out)
        return summarize_module.PaperSummary(**data).model_dump()

    ollama_run = _time_call(_ollama_call)
    if ollama_run["success"]:
        print("  Ollama: " + str(round(ollama_run["elapsed"], 1)) + "s  (ok)")
    else:
        print("  Ollama: " + str(round(ollama_run["elapsed"], 1)) + "s  (FAILED: " + str(ollama_run["error"]) + ")")

    return {
        "paper": name,
        "char_count": len(cleaned),
        "groq": groq_run,
        "ollama": ollama_run,
    }


def write_report(results, out_path):
    lines = [
        "# Groq vs Ollama Benchmark",
        "",
        "Comparing cloud (Groq, `" + summarize_module.GROQ_MODEL + "`) vs local (Ollama, `" + summarize_module.LOCAL_MODEL + "`) summarization on " + str(len(results)) + " real papers.",
        "",
        "| Paper | Chars | Groq time | Groq status | Ollama time | Ollama status |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        g = r["groq"]
        o = r["ollama"]
        g_status = "ok" if g["success"] else "FAILED: " + str(g["error"])[:40]
        o_status = "ok" if o["success"] else "FAILED: " + str(o["error"])[:40]
        lines.append("| " + r["paper"] + " | " + str(r["char_count"]) + " | " + str(round(g["elapsed"],1)) + "s | " + g_status + " | " + str(round(o["elapsed"],1)) + "s | " + o_status + " |")

    groq_times = [r["groq"]["elapsed"] for r in results if r["groq"]["success"]]
    ollama_times = [r["ollama"]["elapsed"] for r in results if r["ollama"]["success"]]
    if groq_times and ollama_times:
        avg_groq = sum(groq_times) / len(groq_times)
        avg_ollama = sum(ollama_times) / len(ollama_times)
        lines.append("")
        lines.append("## Summary")
        lines.append("- Average Groq time: " + str(round(avg_groq,1)) + "s")
        lines.append("- Average Ollama time: " + str(round(avg_ollama,1)) + "s")
        lines.append("- Ollama is ~" + str(round(avg_ollama/avg_groq,1)) + "x slower than Groq on this hardware")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nReport written to " + out_path)


if __name__ == "__main__":
    papers = sorted(glob.glob("tests/sample_papers/*.pdf"))
    print("Found " + str(len(papers)) + " papers to benchmark.")

    results = [benchmark_paper(p) for p in papers]

    os.makedirs("docs", exist_ok=True)
    write_report(results, "docs/benchmark.md")
