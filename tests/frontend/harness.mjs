// Loads the inline <script> out of static/index.html and evaluates it in a
// sandbox, so the frontend's pure helpers can be unit-tested directly.
//
// This exists so the single-file frontend stays single-file: no build step, no
// npm dependency, no splitting the page into modules just to make it testable.
// Anything touching the real DOM is out of scope here — the stub below only
// needs to be good enough that top-level setup code runs without throwing.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const HERE = dirname(fileURLToPath(import.meta.url));
export const INDEX_HTML = join(HERE, "..", "..", "static", "index.html");

/** An object that tolerates any property access, call, or chain. */
function permissive(name = "stub") {
  const target = function () { return permissive(name); };
  target.nodeName = name;
  return new Proxy(target, {
    get(_t, prop) {
      if (prop === Symbol.toPrimitive) return () => "";
      if (prop === "then") return undefined;              // not a thenable
      if (prop === "length" || prop === "offsetWidth") return 0;
      if (prop === "children" || prop === "classList") return permissive(name);
      return permissive(String(prop));
    },
    set() { return true; },
    apply() { return permissive(name); },
    has() { return true; },
  });
}

function makeSandbox() {
  const doc = {
    getElementById: () => permissive("element"),
    querySelector: () => permissive("element"),
    querySelectorAll: () => [],
    createElement: () => permissive("element"),
    addEventListener: () => {},
    body: permissive("body"),
    documentElement: permissive("html"),
  };
  const win = {
    addEventListener: () => {},
    innerWidth: 1280,
    innerHeight: 800,
    matchMedia: () => ({ matches: false, addEventListener: () => {} }),
    localStorage: {
      _v: {},
      getItem(k) { return k in this._v ? this._v[k] : null; },
      setItem(k, v) { this._v[k] = String(v); },
      removeItem(k) { delete this._v[k]; },
    },
    location: { href: "http://localhost:8000/", origin: "http://localhost:8000" },
  };
  const sandbox = {
    window: win, document: doc, navigator: { userAgent: "node" },
    localStorage: win.localStorage, location: win.location,
    console, fetch: async () => ({ ok: true, json: async () => ({}) }),
    setTimeout, clearTimeout, setInterval, clearInterval,
    requestAnimationFrame: (fn) => setTimeout(fn, 0),
    encodeURIComponent, decodeURIComponent, URL, Date, Math, JSON,
    alert: () => {}, Blob: class {}, FileReader: class {},
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  return sandbox;
}

/** Extract the page's main inline script (the one after the stylesheet). */
export function extractInlineScript(html = readFileSync(INDEX_HTML, "utf8")) {
  const from = html.indexOf("<script>", html.indexOf("</style>"));
  if (from === -1) throw new Error("no inline <script> found in index.html");
  const start = from + "<script>".length;
  const end = html.indexOf("</script>", start);
  return html.slice(start, end);
}

/** Evaluate the frontend and hand back its sandbox for poking at. */
export function loadFrontend() {
  const sandbox = makeSandbox();
  const context = vm.createContext(sandbox);
  vm.runInContext(extractInlineScript(), context, { filename: "index.html<script>" });
  return sandbox;
}
