// Unit tests for the frontend's pure helpers. Run with:  node --test tests/
import test from "node:test";
import assert from "node:assert/strict";
import { loadFrontend } from "./harness.mjs";

const fe = loadFrontend();

test("the whole inline script evaluates without throwing", () => {
  // Guards against a syntax error or a top-level crash shipping unnoticed —
  // the page has no build step, so nothing else would catch it.
  assert.equal(typeof fe.renderResults, "function");
  assert.equal(typeof fe.summarizePdf, "function");
});

// ---------------------------------------------------------------------------
// escapeHtml — every render path funnels untrusted model output through this
// ---------------------------------------------------------------------------

test("escapeHtml neutralises the script-injection characters", () => {
  assert.equal(
    fe.escapeHtml('<script>alert("x")</script>'),
    "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
  );
  assert.equal(fe.escapeHtml("a & b"), "a &amp; b");
  assert.equal(fe.escapeHtml("it's"), "it&#39;s");
});

test("escapeHtml escapes the ampersand before the entities it introduces", () => {
  // Wrong order here would double-escape and render literal "&amp;lt;"
  assert.equal(fe.escapeHtml("&lt;"), "&amp;lt;");
});

test("escapeHtml handles null and undefined as empty, not as text", () => {
  assert.equal(fe.escapeHtml(null), "");
  assert.equal(fe.escapeHtml(undefined), "");
  assert.equal(fe.escapeHtml(0), "0");
});

// ---------------------------------------------------------------------------
// normalizeDoi — must agree with normalize_doi() in src/pipeline.py
// ---------------------------------------------------------------------------

test("normalizeDoi pulls the DOI out of the forms users actually paste", () => {
  const want = "10.1371/journal.pone.0121283";
  for (const input of [
    want,
    `  ${want}  `,
    `doi:${want}`,
    `DOI: ${want}`,
    `https://doi.org/${want}`,
    `http://dx.doi.org/${want}`,
    `${want}.`,
    `${want},`,
  ]) {
    assert.equal(fe.normalizeDoi(input), want, `failed on: ${input}`);
  }
});

test("normalizeDoi leaves non-DOI input alone rather than inventing one", () => {
  assert.equal(fe.normalizeDoi("not a doi"), "not a doi");
  assert.equal(fe.normalizeDoi(""), "");
  assert.equal(fe.normalizeDoi(null), "");
});

// ---------------------------------------------------------------------------
// Render helpers — hostile model output must not become markup
// ---------------------------------------------------------------------------

test("renderParas splits on blank lines and escapes each paragraph", () => {
  const html = fe.renderParas("first para\n\nsecond <b>para</b>");
  assert.equal((html.match(/<p class="sec-paragraph">/g) || []).length, 2);
  assert.ok(html.includes("&lt;b&gt;para&lt;/b&gt;"));
  assert.ok(!html.includes("<b>"));
});

test("renderParas drops empty and whitespace-only paragraphs", () => {
  assert.equal(fe.renderParas("\n\n   \n\n"), "");
  assert.equal(fe.renderParas(null), "");
});

test("renderResultsTable escapes every cell", () => {
  const html = fe.renderResultsTable([
    { metric: "<img src=x onerror=alert(1)>", value: "99%", comparison: null },
  ]);
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img"));
  assert.ok(html.includes("—"), "missing comparison should fall back to an em dash");
});

test("renderAccordion escapes its label", () => {
  const html = fe.renderAccordion("acc-1", '</summary><script>x</script>', "<p>body</p>");
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("<p>body</p>"), "inner html is trusted and passed through");
});

// ---------------------------------------------------------------------------
// Citation rendering — the honest-grounding guarantee, at the UI layer
// ---------------------------------------------------------------------------

test("a verified finding renders a clickable page tab", () => {
  const html = fe.renderFindingsList([
    { finding: "F", detail: "D", quote: "a quote", citation: { verified: true, page: 4 } },
  ]);
  assert.ok(html.includes("p. 4"));
  assert.ok(html.includes("jumpToCitation(4"));
  assert.ok(!html.includes("finding-unverified"));
});

test("an unverified finding is visibly marked and not clickable", () => {
  const html = fe.renderFindingsList([
    { finding: "F", citation: { verified: false, page: null } },
  ]);
  assert.ok(html.includes("unverified"));
  assert.ok(html.includes("finding-unverified"));
  assert.ok(!html.includes("jumpToCitation"));
});

test("a finding with no citation at all is treated as unverified", () => {
  // The server always attaches one, but the UI must not fail open if it does not.
  const html = fe.renderFindingsList([{ finding: "F" }]);
  assert.ok(html.includes("unverified"));
  assert.ok(!html.includes("jumpToCitation"));
});

test("finding text and quotes are escaped in the citation tab", () => {
  const html = fe.renderFindingsList([
    { finding: '<script>x</script>', quote: '"><script>y</script>', citation: { verified: true, page: 1 } },
  ]);
  assert.ok(!html.includes("<script>"));
});

// ---------------------------------------------------------------------------
// Page highlighting — sentinel-based, so it must survive hostile page text
// ---------------------------------------------------------------------------

test("renderPageHtml highlights a quote without letting page text become markup", () => {
  fe.window.citeQuotes = [];
  const html = fe.renderPageHtml({ n: 1, text: "before <b>evil</b> the target phrase after" }, "the target phrase");
  assert.ok(html.includes('<mark class="cite-mark">the target phrase</mark>'));
  assert.ok(!html.includes("<b>"));
  assert.ok(html.includes("&lt;b&gt;"));
});

test("renderPageHtml tolerates whitespace differences between quote and page", () => {
  fe.window.citeQuotes = [];
  const html = fe.renderPageHtml({ n: 1, text: "a phrase\nsplit across   lines" }, "phrase split across lines");
  assert.ok(html.includes('<mark class="cite-mark">'));
});

test("renderPageHtml leaves the page alone when the quote is absent", () => {
  fe.window.citeQuotes = [];
  const html = fe.renderPageHtml({ n: 1, text: "nothing to see" }, "not present here");
  assert.ok(!html.includes("<mark"));
  assert.equal(html, "nothing to see");
});

test("renderPageHtml does not let regex metacharacters in a quote throw", () => {
  fe.window.citeQuotes = [];
  assert.doesNotThrow(() => fe.renderPageHtml({ n: 1, text: "cost is $5 (approx.)" }, "$5 (approx.)"));
});

// ---------------------------------------------------------------------------
// Error copy and formatting
// ---------------------------------------------------------------------------

test("friendlyChatError maps provider failures to actionable copy", () => {
  assert.match(fe.friendlyChatError(429, ""), /rate-limited/);
  assert.match(fe.friendlyChatError(200, "tokens per day exceeded"), /rate-limited/);
  assert.match(fe.friendlyChatError(503, ""), /unavailable/);
  assert.match(fe.friendlyChatError(200, "Ollama is not running"), /unavailable/);
  assert.match(fe.friendlyChatError(500, ""), /could not answer/);
});

test("friendlyChatError passes a 400's detail through verbatim", () => {
  assert.equal(fe.friendlyChatError(400, "Question cannot be empty."), "Question cannot be empty.");
  assert.match(fe.friendlyChatError(400, ""), /Invalid request/);
});

test("_timeAgo reads naturally across the ranges", () => {
  const now = Date.now();
  assert.equal(fe._timeAgo(now), "just now");
  assert.equal(fe._timeAgo(now - 90 * 1000), "2m ago");
  assert.equal(fe._timeAgo(now - 3 * 3600 * 1000), "3h ago");
  assert.equal(fe._timeAgo(now - 2 * 86400 * 1000), "2d ago");
});

// ---------------------------------------------------------------------------
// Markdown export
// ---------------------------------------------------------------------------

test("exportReport emits the sections a paper actually has", () => {
  const md = fe.exportReport({
    title: "A Paper", authors: ["Ada", "Grace"], one_line_summary: "It works.",
    overview: "Overview text.", problem_statement: "The problem.", approach: "The approach.",
    key_findings: [{ finding: "Finding one", detail: "42%" }],
    results_table: [{ metric: "Accuracy", value: "99%", comparison: "+3" }],
    significance: "It matters.",
  });
  assert.ok(md.startsWith("# A Paper"));
  assert.ok(md.includes("**Authors:** Ada, Grace"));
  assert.ok(md.includes("> It works."));
  assert.ok(md.includes("## Key findings"));
  assert.ok(md.includes("- Finding one — 42%"));
  assert.ok(md.includes("| Accuracy | 99% | +3 |"));
});

test("exportReport omits absent sections instead of emitting empty headings", () => {
  const md = fe.exportReport({ title: "Bare" });
  assert.ok(md.includes("# Bare"));
  for (const heading of ["## Overview", "## Key findings", "## Results", "## Approach"]) {
    assert.ok(!md.includes(heading), `should not emit ${heading}`);
  }
});

test("exportReport falls back to a title rather than emitting an empty one", () => {
  assert.ok(fe.exportReport({}).startsWith("# Untitled paper"));
});
