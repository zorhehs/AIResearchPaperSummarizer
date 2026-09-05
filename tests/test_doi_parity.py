"""The DOI normaliser exists twice — normalize_doi() in src/pipeline.py and
normalizeDoi() in static/index.html. The frontend cleans a pasted DOI before
sending it and the backend cleans it again on arrival, so the two have to agree:
if they drift, a DOI the UI accepts can be rejected by the server (or vice
versa) for reasons no error message would explain.

These tests run the same corpus through both implementations and compare.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pipeline import normalize_doi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "frontend", "harness.mjs")

CASES = [
    "10.1371/journal.pone.0121283",
    "  10.1371/journal.pone.0121283  ",
    "doi:10.1371/journal.pone.0121283",
    "DOI: 10.1371/journal.pone.0121283",
    "https://doi.org/10.1371/journal.pone.0121283",
    "http://dx.doi.org/10.1016/j.cell.2015.05.001",
    "https://www.doi.org/10.1234/abc",
    "10.1371/journal.pone.0121283.",
    "10.1371/journal.pone.0121283,",
    "10.1371/journal.pone.0121283;",
    "10.1371/journal.pone.0121283...",
    "not a doi",
    "",
    "   ",
    "See 10.1234/abc.def in the paper",
    "10.1234/abc(def)",
    "DOI 10.1234/x-y_z",
    "10.1234/ABC.DEF",
    "\t10.5555/12345678\n",
]

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _normalize_in_js(cases):
    script = (
        f"import {{ loadFrontend }} from {json.dumps(HARNESS)};\n"
        "const fe = loadFrontend();\n"
        "console.log(JSON.stringify(JSON.parse(process.argv[1]).map(c => fe.normalizeDoi(c))));\n"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script, "--", json.dumps(cases)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"node failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


@needs_node
def test_frontend_and_backend_normalise_dois_identically():
    js_results = _normalize_in_js(CASES)
    mismatches = [
        (case, normalize_doi(case), js)
        for case, js in zip(CASES, js_results)
        if normalize_doi(case) != js
    ]
    assert not mismatches, "normalizeDoi/normalize_doi disagree on:\n" + "\n".join(
        f"  {c!r}: python={p!r} js={j!r}" for c, p, j in mismatches
    )


def test_backend_normaliser_handles_the_corpus_without_raising():
    # Useful on its own when node is unavailable.
    for case in CASES:
        assert isinstance(normalize_doi(case), str)
