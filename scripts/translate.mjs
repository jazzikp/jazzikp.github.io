#!/usr/bin/env node
/**
 * Translate the site into Chinese with Grok, at authoring time.
 *
 * The translation is generated once, committed, and reviewed like any other
 * content. Readers get both languages in the HTML and the toggle is instant:
 * no API call on page view, nothing to rate limit, and the service worker can
 * still serve the page offline.
 *
 * Two kinds of text are handled:
 *
 *   posts  — the body of each `_posts/*.md`, rewritten into the EN/ZH panel
 *            pair that the toggle in js/site.js shows and hides.
 *   ui     — the short strings in `_data/i18n.yml`: navigation, footer, and
 *            the headings on the pages that are not posts.
 *
 * A translated post must round-trip its own structure. Every fenced code
 * block, `$$` maths delimiter, link target and image source in the English
 * body has to survive into the Chinese one, or the file is not written. A
 * model that quietly drops a backslash inside `$$…$$` produces maths that
 * renders as plain text, and that failure is silent in the browser — so it is
 * caught here instead.
 *
 * Run:
 *   XAI_API_KEY=… node scripts/translate.mjs            # everything missing
 *   XAI_API_KEY=… node scripts/translate.mjs --only _posts/2026-08-16-x.md
 *   node scripts/translate.mjs --check                  # report gaps, no API
 */
import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const postsDir = join(root, "_posts");
const uiFile = join(root, "_data", "i18n.yml");

const API = "https://api.x.ai/v1/chat/completions";
const MODEL = "grok-4.6";

const PROSE_SYSTEM = `You translate English technical writing into Simplified Chinese for a machine-learning blog.

Rules, in order of importance:

1. Return the document and nothing else. No preamble, no code fence around the whole answer, no notes about what you did.
2. Keep every fenced code block byte-for-byte identical, including the language tag, the code inside, and the comments inside. Do not translate code or comments in code.
3. Keep every maths expression byte-for-byte identical. Anything between $$ and $$ is maths. Do not translate it, do not reformat it, and never remove a backslash.
4. Keep the Markdown structure identical: the same headings at the same levels, the same list shapes, the same block quotes, the same tables, the same kramdown attribute lists such as {: loading="lazy"}.
5. Keep every link and image target identical. Translate link text, never the URL.
6. Leave established technical terms in English where a Chinese reader in this field would expect them: Transformer, Pre-Norm, Post-Norm, LayerNorm, RMSNorm, attention, residual, embedding, logits, softmax, tokenizer, JAX, GPU, and the names of papers, models and libraries.
7. Write the way a working Chinese ML engineer writes: direct, unpadded, no marketing register.`;

const UI_SYSTEM = `You translate short user-interface strings for a machine-learning blog into Simplified Chinese.

Return only a JSON object mapping each input key to its translated string. No other text.

Keep translations short — these are navigation links, buttons and headings, and they have to fit the same space as the English. Leave product and section names that a Chinese reader would expect in English as they are.`;

function parseArgs(argv) {
  const args = { check: false, force: false, only: null, ui: true, posts: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--check") args.check = true;
    else if (a === "--force") args.force = true;
    else if (a === "--only") args.only = argv[++i];
    else if (a === "--posts-only") args.ui = false;
    else if (a === "--ui-only") args.posts = false;
    else throw new Error(`Unknown argument: ${a}`);
  }
  return args;
}

async function grok(system, user) {
  const key = process.env.XAI_API_KEY;
  if (!key) throw new Error("XAI_API_KEY is not set");

  const res = await fetch(API, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });

  if (!res.ok) {
    throw new Error(`xAI ${res.status}: ${(await res.text()).slice(0, 400)}`);
  }
  const body = await res.json();
  const text = body.choices?.[0]?.message?.content;
  if (!text) throw new Error("xAI returned no content");
  return text.trim();
}

// ---------------------------------------------------------------------------
// Structure checks
// ---------------------------------------------------------------------------

const FENCE = /^[ \t]*(```|~~~)/gm;
const MATH = /\$\$/g;
const LINK = /\]\(([^)\s]+)/g;
const IMAGE = /!\[[^\]]*\]\(([^)\s]+)/g;
const HEADING = /^#{1,6} /gm;

function count(text, re) {
  return (text.match(re) || []).length;
}

function targets(text, re) {
  return [...text.matchAll(re)].map((m) => m[1]).sort();
}

/** Everything that must be identical on both sides of the translation. */
function shape(text) {
  return {
    fences: count(text, FENCE),
    math: count(text, MATH),
    headings: count(text, HEADING),
    links: targets(text, LINK),
    images: targets(text, IMAGE),
  };
}

function compareShape(english, chinese) {
  const a = shape(english);
  const b = shape(chinese);
  const problems = [];

  if (a.fences !== b.fences) problems.push(`code fences ${a.fences} → ${b.fences}`);
  if (a.math !== b.math) problems.push(`$$ delimiters ${a.math} → ${b.math}`);
  if (a.math % 2 !== 0 || b.math % 2 !== 0) problems.push("unbalanced $$ delimiters");
  if (a.headings !== b.headings) problems.push(`headings ${a.headings} → ${b.headings}`);
  if (String(a.links) !== String(b.links)) {
    const lost = a.links.filter((l) => !b.links.includes(l));
    const gained = b.links.filter((l) => !a.links.includes(l));
    problems.push(`link targets changed (lost ${lost.length}, gained ${gained.length})`);
  }
  if (String(a.images) !== String(b.images)) problems.push("image sources changed");

  // A translation that is still English has not failed any structural check.
  const han = (chinese.match(/[\u4e00-\u9fff]/g) || []).length;
  if (han < 50) problems.push(`only ${han} Chinese characters — looks untranslated`);

  return problems;
}

// ---------------------------------------------------------------------------
// Posts
// ---------------------------------------------------------------------------

const FRONT_MATTER = /^---\n([\s\S]*?)\n---\n?/;
const KRAMDOWN_OPTIONS = /^\{::options[^}]*\/\}\s*/m;

function splitPost(raw) {
  const m = raw.match(FRONT_MATTER);
  if (!m) throw new Error("no front matter");
  return { frontMatter: m[1], body: raw.slice(m[0].length) };
}

function frontMatterValue(frontMatter, key) {
  const m = frontMatter.match(new RegExp(`^${key}:\\s*(.+)$`, "m"));
  if (!m) return null;
  return m[1].trim().replace(/^["']|["']$/g, "");
}

function isBilingual(frontMatter) {
  return /^bilingual:\s*true\s*$/m.test(frontMatter);
}

/** Insert a key after an existing one so the front matter stays readable. */
function insertAfter(frontMatter, afterKey, line) {
  const re = new RegExp(`^(${afterKey}:.*)$`, "m");
  if (!re.test(frontMatter)) return `${frontMatter}\n${line}`;
  return frontMatter.replace(re, `$1\n${line}`);
}

function yamlQuote(value) {
  return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function buildBilingualBody(english, chinese) {
  // The kramdown directive has to stay at the top level, outside the panels,
  // or the markdown inside the divs is emitted as literal text.
  const directive = '{::options parse_block_html="true" /}';
  const en = english.replace(KRAMDOWN_OPTIONS, "").trim();
  const zh = chinese.replace(KRAMDOWN_OPTIONS, "").trim();

  return [
    "",
    directive,
    "",
    '<div data-lang-panel="en" markdown="1">',
    "",
    en,
    "",
    "</div>",
    "",
    '<div data-lang-panel="zh" hidden markdown="1">',
    "",
    zh,
    "",
    "</div>",
    "",
  ].join("\n");
}

async function translatePost(file, args) {
  const rel = relative(root, file);
  const raw = await readFile(file, "utf8");
  const { frontMatter, body } = splitPost(raw);

  if (isBilingual(frontMatter) && !args.force) {
    return { rel, status: "already bilingual" };
  }
  if (args.check) {
    return { rel, status: "needs translation" };
  }

  const title = frontMatterValue(frontMatter, "title");
  const subtitle = frontMatterValue(frontMatter, "subtitle");

  const english = body.replace(KRAMDOWN_OPTIONS, "").trim();
  const chinese = await grok(PROSE_SYSTEM, english);

  const problems = compareShape(english, chinese);
  if (problems.length) {
    return { rel, status: "REJECTED", problems };
  }

  const strings = await grok(
    UI_SYSTEM,
    JSON.stringify({ title, ...(subtitle ? { subtitle } : {}) })
  );
  let parsed = {};
  try {
    parsed = JSON.parse(strings.replace(/^```(?:json)?\s*|\s*```$/g, ""));
  } catch {
    return { rel, status: "REJECTED", problems: ["title translation was not JSON"] };
  }

  let fm = frontMatter;
  if (parsed.title) fm = insertAfter(fm, "title", `title_zh: ${yamlQuote(parsed.title)}`);
  if (subtitle && parsed.subtitle) {
    fm = insertAfter(fm, "subtitle", `subtitle_zh: ${yamlQuote(parsed.subtitle)}`);
  }
  if (!isBilingual(fm)) fm = insertAfter(fm, "date", "bilingual: true");

  await writeFile(file, `---\n${fm}\n---\n${buildBilingualBody(english, chinese)}`, "utf8");
  return { rel, status: "translated" };
}

// ---------------------------------------------------------------------------
// Interface strings
// ---------------------------------------------------------------------------

/**
 * `_data/i18n.yml` is a flat two-level map of `key: {en, zh}`. It is small and
 * fully under our control, so it is read and written with a narrow parser
 * rather than adding a YAML dependency to a repo that has none.
 */
function parseI18n(text) {
  const out = {};
  let section = null;
  for (const line of text.split("\n")) {
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const top = line.match(/^([a-z0-9_]+):\s*$/i);
    if (top) {
      section = top[1];
      out[section] = {};
      continue;
    }
    const entry = line.match(/^ {2}([a-z0-9_]+):\s*$/i);
    if (entry && section) {
      out[section][entry[1]] = { en: null, zh: null };
      continue;
    }
    const leaf = line.match(/^ {4}(en|zh):\s*(.*)$/);
    if (leaf && section) {
      const keys = Object.keys(out[section]);
      const last = keys[keys.length - 1];
      const value = leaf[2].trim().replace(/^["']|["']$/g, "");
      out[section][last][leaf[1]] = value === "" ? null : value;
    }
  }
  return out;
}

function serializeI18n(data, header) {
  const lines = [header.trimEnd(), ""];
  for (const [section, entries] of Object.entries(data)) {
    lines.push(`${section}:`);
    for (const [key, value] of Object.entries(entries)) {
      lines.push(`  ${key}:`);
      lines.push(`    en: ${yamlQuote(value.en ?? "")}`);
      lines.push(`    zh: ${value.zh ? yamlQuote(value.zh) : '""'}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

async function translateUi(args) {
  let text;
  try {
    text = await readFile(uiFile, "utf8");
  } catch {
    return { rel: relative(root, uiFile), status: "missing" };
  }

  const header = text.split(/^\w+:/m)[0];
  const data = parseI18n(text);

  const missing = [];
  for (const [section, entries] of Object.entries(data)) {
    for (const [key, value] of Object.entries(entries)) {
      if (value.en && !value.zh) missing.push({ section, key, en: value.en });
    }
  }

  if (!missing.length) return { rel: relative(root, uiFile), status: "complete" };
  if (args.check) {
    return { rel: relative(root, uiFile), status: `${missing.length} strings missing` };
  }

  const payload = {};
  for (const m of missing) payload[`${m.section}.${m.key}`] = m.en;

  const answer = await grok(UI_SYSTEM, JSON.stringify(payload, null, 2));
  let parsed;
  try {
    parsed = JSON.parse(answer.replace(/^```(?:json)?\s*|\s*```$/g, ""));
  } catch {
    return { rel: relative(root, uiFile), status: "REJECTED", problems: ["not JSON"] };
  }

  let filled = 0;
  for (const m of missing) {
    const value = parsed[`${m.section}.${m.key}`];
    if (typeof value === "string" && value.trim()) {
      data[m.section][m.key].zh = value.trim();
      filled++;
    }
  }

  await writeFile(uiFile, serializeI18n(data, header), "utf8");
  return { rel: relative(root, uiFile), status: `${filled} of ${missing.length} translated` };
}

// ---------------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const results = [];

  if (args.posts) {
    let files;
    if (args.only) {
      files = [resolve(root, args.only)];
    } else {
      files = (await readdir(postsDir))
        .filter((f) => f.endsWith(".md"))
        .map((f) => join(postsDir, f))
        .sort();
    }
    for (const file of files) {
      try {
        results.push(await translatePost(file, args));
      } catch (error) {
        results.push({ rel: relative(root, file), status: "ERROR", problems: [error.message] });
      }
    }
  }

  if (args.ui && !args.only) {
    try {
      results.push(await translateUi(args));
    } catch (error) {
      results.push({ rel: "_data/i18n.yml", status: "ERROR", problems: [error.message] });
    }
  }

  let bad = 0;
  for (const r of results) {
    console.log(`${r.status.padEnd(22)} ${r.rel}`);
    for (const p of r.problems || []) console.log(`    ${p}`);
    if (r.status === "REJECTED" || r.status === "ERROR") bad++;
  }

  if (bad) {
    console.error(`\n${bad} item(s) failed. Nothing was written for those.`);
    process.exitCode = 1;
  }
}

await main();
