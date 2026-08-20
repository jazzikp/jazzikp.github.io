// Fetches the Jekyll-built /corpus.json from the live site and picks the
// excerpts that belong in the chat system prompt. New posts show up after
// the next GitHub Pages build — no worker redeploy, no prompt rewrite.
const DEFAULT_URL = "https://jazzikp.github.io/corpus.json";
const TTL_MS = 60 * 60 * 1000;
const MAX_EXCERPT = 3500;
const MAX_DOCS = 6;

let cache = { at: 0, data: null };

export async function loadCorpus(env) {
  const url = (env && env.CORPUS_URL) || DEFAULT_URL;
  if (cache.data && Date.now() - cache.at < TTL_MS) return cache.data;
  try {
    const res = await fetch(url, { cf: { cacheTtl: 3600 } });
    if (!res.ok) return cache.data;
    const data = await res.json();
    cache = { at: Date.now(), data };
    return data;
  } catch {
    return cache.data;
  }
}

function tokens(s) {
  return [...new Set(String(s || "").toLowerCase().match(/[a-z][a-z0-9+_-]{1,}|[\u4e00-\u9fff]+|\d{4}/g) || [])];
}

function hayTokens(doc) {
  return tokens(
    [doc.title, doc.title_zh, doc.subtitle, doc.subtitle_zh, (doc.tags || []).join(" "), doc.kind, doc.text]
      .filter(Boolean)
      .join(" ")
  );
}

function scoreDoc(doc, q) {
  const hay = new Set(hayTokens(doc));
  let s = 0;
  for (const t of q) {
    if (hay.has(t)) s += t.length >= 4 ? 3 : 2;
  }
  if (doc.kind === "about" || doc.kind === "home") s += 1;
  return s;
}

function allDocs(corpus) {
  const out = [];
  for (const d of corpus.pages || []) out.push(d);
  for (const d of corpus.posts || []) out.push(d);
  for (const d of corpus.projects || []) out.push(d);
  for (const d of corpus.reports || []) out.push(d);
  for (const d of corpus.upcoming_reports || []) out.push(d);
  return out;
}

export function formatSiteContext(corpus, userText) {
  if (!corpus) return "";
  const docs = allDocs(corpus);
  if (!docs.length) return "";
  const q = tokens(userText);
  const catalog = docs
    .map((d) => {
      const when = d.date || d.year || d.eta || "";
      const zh = d.title_zh ? " / " + d.title_zh : "";
      const url = d.url ? " (" + d.url + ")" : "";
      return "- " + (when ? when + " · " : "") + d.title + zh + url;
    })
    .join("\n");

  const ranked = docs.map((d) => ({ d, s: scoreDoc(d, q) })).sort((a, b) => b.s - a.s);
  const selected = [];
  const seen = new Set();
  for (const d of docs) {
    if (d.kind === "about" || d.kind === "home") {
      selected.push(d);
      seen.add(d.url || d.title);
    }
  }
  for (const { d, s } of ranked) {
    if (selected.length >= MAX_DOCS) break;
    const key = d.url || d.title;
    if (seen.has(key)) continue;
    if (q.length && s <= 0) continue;
    selected.push(d);
    seen.add(key);
  }
  if (selected.length < 3) {
    for (const d of docs) {
      if (selected.length >= 4) break;
      const key = d.url || d.title;
      if (seen.has(key)) continue;
      if (d.kind === "post") {
        selected.push(d);
        seen.add(key);
      }
    }
  }

  const excerpts = selected
    .map((d) => {
      const head = [d.title, d.date || d.year, d.url].filter(Boolean).join(" · ");
      const body = String(d.text || d.subtitle || "").slice(0, MAX_EXCERPT);
      return "## " + head + "\n" + body;
    })
    .join("\n\n");

  return (
    "SITE CATALOG (every published page and post; use this list, do not invent titles):\n" +
    catalog +
    "\n\nSITE EXCERPTS (answer from these; if it is not here, say you have not written it up):\n" +
    excerpts
  );
}

export async function siteContextFor(messages, env) {
  const corpus = await loadCorpus(env);
  const lastUser = [...(messages || [])].reverse().find((m) => m && m.role === "user");
  return formatSiteContext(corpus, lastUser ? lastUser.content : "");
}
