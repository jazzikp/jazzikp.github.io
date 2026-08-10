/*
 * Nothing on the site points at something that is not there.
 *
 * Crawls every internal link from the home page outward and resolves every
 * asset reference it finds, including each candidate in a srcset.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startSite } from "./helpers/site.mjs";

let site;
before(async () => { site = await startSite(); });
after(async () => { await site.close(); });

const isPage = (p) => p.endsWith("/") || p.endsWith(".html");
const isExternal = (href) => /^(https?:|mailto:|tel:|data:|#)/.test(href);

async function crawl(origin) {
  const seen = new Map();
  const pages = new Set(["/"]);
  const queue = ["/"];
  const broken = [];

  const check = async (path, from) => {
    if (seen.has(path)) return seen.get(path);
    const res = await fetch(origin + path);
    seen.set(path, res.status);
    if (res.status !== 200) broken.push({ path, status: res.status, from });
    return res.status;
  };

  while (queue.length) {
    const page = queue.shift();
    const res = await fetch(origin + page);
    seen.set(page, res.status);
    if (!res.ok) { broken.push({ path: page, status: res.status, from: "(entry)" }); continue; }
    const html = await res.text();

    const refs = [...html.matchAll(/(?:href|src)="([^"]+)"/g)].map((m) => m[1]);
    const srcset = [...html.matchAll(/srcset="([^"]+)"/g)].flatMap((m) =>
      m[1].split(",").map((c) => c.trim().split(/\s+/)[0])
    );

    for (const ref of [...refs, ...srcset]) {
      if (isExternal(ref) || !ref) continue;
      const path = new URL(ref, origin + page).pathname;
      if (isPage(path)) {
        if (!pages.has(path)) { pages.add(path); queue.push(path); }
      } else {
        await check(path, page);
      }
    }
  }

  return { pages, checked: seen, broken };
}

describe("links and assets", () => {
  test("no internal link or asset is broken", async () => {
    const { pages, checked, broken } = await crawl(site.origin);
    assert.ok(pages.size >= 9, `only reached ${pages.size} pages — the crawl looks stuck`);
    assert.ok(checked.size >= 25, "suspiciously few references checked");
    assert.deepEqual(
      broken,
      [],
      "broken references:\n" +
        broken.map((b) => `  ${b.status} ${b.path}  (linked from ${b.from})`).join("\n")
    );
  });

  test("unknown URLs serve the 404 page", async () => {
    const res = await fetch(site.origin + "/definitely-not-a-page/");
    assert.equal(res.status, 404);
    assert.match(await res.text(), /drifted off the map/);
  });

  test("the old /tags/ URL still redirects to /blogs/", async () => {
    const html = await (await fetch(site.origin + "/tags/")).text();
    assert.match(html, /\/blogs\//);
    assert.match(html, /http-equiv="refresh"/);
  });
});
