/*
 * Maths must actually typeset.
 *
 * The trap this guards against: kramdown treats a backslash before a
 * punctuation character as a Markdown escape, so writing \(x\) in a post gets
 * the backslashes eaten and MathJax is handed a plain "(x)" that it never
 * recognises as maths. It fails silently — the build is clean, the page is
 * clean, and raw LaTeX just sits there in the prose.
 *
 * The supported syntax is $$...$$. Kramdown parses it as maths and emits the
 * delimiter MathJax wants, choosing inline or display from the context.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { startSite, ROOT, POSTS } from "./helpers/site.mjs";

let site;
const sources = new Map();
const html = new Map();

before(async () => {
  site = await startSite();
  const dir = join(ROOT, "_posts");
  for (const name of await readdir(dir)) {
    if (name.endsWith(".md")) sources.set(name, await readFile(join(dir, name), "utf8"));
  }
  for (const page of POSTS) {
    html.set(page, await (await fetch(site.origin + page)).text());
  }
});
after(async () => { await site.close(); });

/** Strip code, which is allowed to contain anything that looks like LaTeX. */
function withoutCode(text) {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/~~~[\s\S]*?~~~/g, " ")
    .replace(/`[^`\n]*`/g, " ");
}

describe("maths", () => {
  test("posts use $$…$$ rather than raw MathJax delimiters", () => {
    const offenders = [];
    for (const [name, source] of sources) {
      const prose = withoutCode(source);
      // \(x\) and \[x\]: kramdown strips the backslashes and the maths is lost.
      for (const [match] of prose.matchAll(/\\[([][\s\S]{0,60}?\\[)\]]/g)) {
        offenders.push(`${name}: ${match.replace(/\s+/g, " ").slice(0, 50)}`);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      "raw MathJax delimiters found — use $$…$$ instead, kramdown converts it:\n  " +
        offenders.join("\n  ")
    );
  });

  test("no raw LaTeX leaks into the rendered prose", () => {
    // Anything MathJax will pick up is removed first: its own \( \) and \[ \]
    // delimiters, plus kramdown's script tags. What is left should read as
    // ordinary prose, with no stray control sequences.
    const leaks = [];
    for (const [page, doc] of html) {
      let prose = doc.slice(doc.indexOf('class="prose'));
      prose = prose
        .replace(/<pre[\s\S]*?<\/pre>/g, " ")
        .replace(/<code[\s\S]*?<\/code>/g, " ")
        .replace(/<script[\s\S]*?<\/script>/g, " ")
        .replace(/\\\([\s\S]*?\\\)/g, " ")   // inline maths MathJax will render
        .replace(/\\\[[\s\S]*?\\\]/g, " ");  // display maths MathJax will render

      const commands = [...prose.matchAll(/\\(?:[a-zA-Z]{2,})/g)].map((m) => m[0]);
      if (commands.length) {
        leaks.push(`${page}: ${[...new Set(commands)].slice(0, 6).join(" ")}`);
      }
      // A subscript or superscript brace outside maths means a delimiter was lost.
      const stranded = [...prose.matchAll(/[_^]\{[^}\n]{1,20}\}/g)].map((m) => m[0]);
      if (stranded.length) {
        leaks.push(`${page}: stranded ${[...new Set(stranded)].slice(0, 4).join(" ")}`);
      }
    }
    assert.deepEqual(
      leaks,
      [],
      "unrendered LaTeX in the prose:\n  " + leaks.join("\n  ")
    );
  });

  test("posts containing maths load MathJax", () => {
    for (const [page, doc] of html) {
      const hasMaths = /\\\(|\\\[/.test(doc.slice(doc.indexOf('class="prose')));
      if (!hasMaths) continue;
      assert.match(doc, /MathJax\.js/, `${page} has maths but never loads MathJax`);
      const tag = doc.match(/<script[^>]*MathJax\.js[^>]*>/)[0];
      assert.match(tag, /\basync\b|\bdefer\b/, `${page} loads MathJax render-blocking: ${tag}`);
    }
  });

  test("every maths expression in a post survives to the page", () => {
    // Counts what the author wrote against what MathJax will be handed, so a
    // silently dropped expression shows up as a mismatch rather than as a
    // gap someone has to notice by eye.
    for (const [page, doc] of html) {
      const name = page.match(/([^/]+)\/$/)[1];
      const source = [...sources.entries()].find(([f]) =>
        f.toLowerCase().includes(name.toLowerCase())
      )?.[1];
      if (!source) continue;

      const written = (withoutCode(source).match(/\$\$/g) || []).length / 2;
      if (!written) continue;

      const prose = doc.slice(doc.indexOf('class="prose'));
      const rendered =
        (prose.match(/\\\([\s\S]*?\\\)/g) || []).length +
        (prose.match(/\\\[[\s\S]*?\\\]/g) || []).length;

      assert.equal(
        rendered,
        written,
        `${page}: ${written} maths expressions written, ${rendered} reached the page`
      );
    }
  });
});
