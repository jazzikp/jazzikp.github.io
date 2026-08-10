/*
 * Performance budgets.
 *
 * Measured from the built files, so the numbers are deterministic and these
 * tests never flake on a slow runner. Budgets live in budgets.json.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { startSite, SITE, PAGES, tags } from "./helpers/site.mjs";

const budgets = JSON.parse(await readFile(new URL("./budgets.json", import.meta.url), "utf8"));
const kb = (n) => `${(n / 1024).toFixed(1)} KB`;

let site;
const html = new Map();

before(async () => {
  site = await startSite();
  for (const page of PAGES) {
    html.set(page, await (await fetch(site.origin + page)).text());
  }
});
after(async () => { await site.close(); });

const sizeOf = async (relative) => (await stat(join(SITE, relative))).size;

describe("performance budgets", () => {
  test("the home page critical path fits its budget", async () => {
    const doc = html.get("/");
    const parts = [["index.html", Buffer.byteLength(doc)]];

    const css = doc.match(/<link rel="stylesheet" href="([^"?]+)/)?.[1];
    parts.push([css, await sizeOf(css)]);

    for (const [, href] of doc.matchAll(/<link rel="preload" as="font"[^>]*href="([^"]+)"/g)) {
      parts.push([href, await sizeOf(href)]);
    }

    // The largest-contentful-paint candidate: the smallest hero srcset entry.
    const hero = doc.match(/srcset="([^"]+)"/)?.[1].split(",")[0].trim().split(/\s+/)[0];
    parts.push([hero, await sizeOf(hero)]);

    const total = parts.reduce((a, [, n]) => a + n, 0);
    assert.ok(
      total <= budgets.criticalPathBytes,
      `critical path is ${kb(total)}, budget ${kb(budgets.criticalPathBytes)}\n` +
        parts.map(([f, n]) => `    ${kb(n).padStart(9)}  ${f}`).join("\n")
    );
  });

  test("the stylesheet fits its budget", async () => {
    const size = await sizeOf("css/site.css");
    assert.ok(size <= budgets.stylesheetBytes, `css/site.css is ${kb(size)}`);
  });

  test("only one script runs on every page, and it is deferred", async () => {
    for (const [page, doc] of html) {
      const scripts = tags(doc, "script").filter((s) => /\bsrc=/.test(s));
      const blocking = scripts.filter((s) => !/\b(defer|async)\b/.test(s));
      assert.deepEqual(
        blocking,
        [],
        `${page} has a render-blocking script: ${blocking.join(", ")}`
      );
    }

    const eager = tags(html.get("/"), "script")
      .filter((s) => /\bsrc="\/js\//.test(s))
      .map((s) => s.match(/src="([^"?]+)/)[1]);
    assert.deepEqual(eager, ["/js/site.js"], "more than site.js is loading up front");

    const size = await sizeOf("js/site.js");
    assert.ok(size <= budgets.firstLoadScriptBytes, `js/site.js is ${kb(size)}`);
  });

  test("the heavy widgets are not on the initial page load", () => {
    for (const [page, doc] of html) {
      for (const lazy of ["chat.js", "comments.js"]) {
        const eagerTag = tags(doc, "script").find(
          (s) => s.includes(lazy) && /\bsrc=/.test(s)
        );
        assert.equal(eagerTag, undefined, `${page} loads ${lazy} eagerly`);
      }
    }
  });

  test("no render-blocking third-party resources", () => {
    for (const [page, doc] of html) {
      const stylesheets = [...doc.matchAll(/<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"/g)]
        .map((m) => m[1])
        .filter((h) => /^https?:/.test(h));
      assert.deepEqual(stylesheets, [], `${page} loads a third-party stylesheet: ${stylesheets}`);

      const external = tags(doc, "script")
        .filter((s) => /src="https?:/.test(s))
        .map((s) => ({ tag: s, host: new URL(s.match(/src="([^"]+)"/)[1]).host }));

      for (const { tag, host } of external) {
        assert.ok(
          budgets.allowedDeferredThirdParty.includes(host),
          `${page} loads an unexpected third-party script from ${host}`
        );
        assert.match(tag, /\basync\b|\bdefer\b/, `${page} third-party script is blocking: ${tag}`);
      }
    }
  });

  test("fonts are self-hosted", () => {
    for (const [page, doc] of html) {
      assert.ok(!doc.includes("fonts.googleapis.com"), `${page} still calls Google Fonts`);
      assert.ok(!doc.includes("fonts.gstatic.com"), `${page} still calls gstatic`);
    }
    const preloads = [...html.get("/").matchAll(/<link rel="preload" as="font"[^>]*>/g)];
    assert.ok(preloads.length >= 1 && preloads.length <= 4, "preload no more fonts than render above the fold");
    for (const [tag] of preloads) {
      assert.match(tag, /crossorigin/, `font preload without crossorigin is fetched twice: ${tag}`);
    }
  });

  test("no shipped asset blows its size budget", async () => {
    const overweight = [];
    const walk = async (dir) => {
      for (const entry of await readdir(join(SITE, dir), { withFileTypes: true })) {
        const rel = join(dir, entry.name);
        if (entry.isDirectory()) { await walk(rel); continue; }
        const size = (await stat(join(SITE, rel))).size;
        const limit = /\.(webp|jpe?g|png|ico)$/i.test(entry.name)
          ? budgets.maxImageBytes
          : /\.woff2$/i.test(entry.name)
            ? budgets.maxFontBytes
            : /\.html$/i.test(entry.name)
              ? budgets.maxHtmlBytes
              : Infinity;
        if (size > limit) overweight.push(`${rel} is ${kb(size)} (limit ${kb(limit)})`);
      }
    };
    await walk(".");
    assert.deepEqual(overweight, [], "assets over budget:\n  " + overweight.join("\n  "));
  });

  test("the whole published site stays small", async () => {
    let total = 0;
    const walk = async (dir) => {
      for (const entry of await readdir(join(SITE, dir), { withFileTypes: true })) {
        const rel = join(dir, entry.name);
        if (entry.isDirectory()) await walk(rel);
        else total += (await stat(join(SITE, rel))).size;
      }
    };
    await walk(".");
    assert.ok(total < 4 * 1024 * 1024, `_site is ${kb(total)} — something large slipped in`);
  });
});
