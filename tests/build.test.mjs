/*
 * The build produces what the site depends on.
 *
 * These catch the failure mode where a config change silently stops emitting a
 * file — the site still builds, and the missing piece only shows up in
 * production.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { startSite, SITE, PAGES } from "./helpers/site.mjs";

let site;
before(async () => { site = await startSite(); });
after(async () => { await site.close(); });

describe("build output", () => {
  test("every page returns 200", async () => {
    for (const page of PAGES) {
      const res = await fetch(site.origin + page);
      assert.equal(res.status, 200, `${page} returned ${res.status}`);
    }
  });

  test("the generated files the site relies on all exist", async () => {
    const required = [
      "css/site.css",
      "js/site.js",
      "js/chat.js",
      "js/comments.js",
      "js/post.js",
      "js/blogs.js",
      "sw.js",
      "feed.xml",
      "sitemap.xml",
      "robots.txt",
      "pwa/manifest.json",
      "img/social-card.jpg",
      "img/favicon-32.png",
      "img/apple-touch-icon.png",
      "corpus.json",
    ];
    for (const file of required) {
      const res = await fetch(site.origin + "/" + file);
      assert.equal(res.status, 200, `missing build output: ${file}`);
    }
  });

  test("the stylesheet is compiled and minified", async () => {
    const css = await readFile(join(SITE, "css/site.css"), "utf8");
    assert.ok(css.includes("@font-face"), "self-hosted fonts are missing from the bundle");
    assert.ok(css.includes(".highlight"), "syntax highlighting is missing from the bundle");
    assert.ok(css.includes("--accent"), "design tokens are missing from the bundle");
    assert.ok(
      css.split("\n").length <= 3,
      "css/site.css is not minified — check `sass: style: compressed` in _config.yml"
    );
  });

  test("Sass partials are not published on their own", async () => {
    // _sass/ must stay private; shipping it would double the CSS payload.
    const res = await fetch(site.origin + "/_sass/_tokens.scss");
    assert.equal(res.status, 404);
  });

  test("full-resolution originals are excluded from the build", async () => {
    const res = await fetch(site.origin + "/img/src/hero-anime.jpg");
    assert.equal(
      res.status,
      404,
      "img/src is shipping — it should be in `exclude` in _config.yml"
    );
  });

  test("the service worker is versioned and wired to the pages", async () => {
    const sw = await readFile(join(SITE, "sw.js"), "utf8");
    const version = sw.match(/const VERSION = "(v\d+)"/)?.[1];
    assert.ok(version, "sw.js has no VERSION — the Liquid front matter may be missing");

    const home = await readFile(join(SITE, "index.html"), "utf8");
    const assetVersion = home.match(/css\/site\.css\?v=(\d+)/)?.[1];
    assert.equal(
      version,
      "v" + assetVersion,
      "sw.js cache version and the asset_version on the pages disagree"
    );

    // A precached URL that 404s silently disables offline support: the install
    // step swallows the error and the shell is never cached.
    const block = sw.match(/const PRECACHE = \[([\s\S]*?)\];/)?.[1];
    assert.ok(block, "could not find the PRECACHE array in sw.js");

    const constants = Object.fromEntries(
      [...sw.matchAll(/^const (\w+) = "([^"]+)";$/gm)].map((m) => [m[1], m[2]])
    );
    const precache = block
      .split(",")
      .map((entry) => entry.replace(/\/\/.*$/gm, "").trim())
      .filter(Boolean)
      .map((entry) => {
        const literal = entry.match(/^"(.*)"$/);
        return literal ? literal[1] : constants[entry];
      });

    assert.ok(precache.length >= 8, `sw.js precache list looks short: ${precache.length}`);
    assert.ok(
      precache.every(Boolean),
      "an entry in PRECACHE is neither a string literal nor a known constant"
    );
    for (const url of precache) {
      const res = await fetch(site.origin + url);
      assert.equal(res.status, 200, `sw.js precaches a missing file: ${url}`);
    }
  });

  test("the kill switch is off", async () => {
    const sw = await readFile(join(SITE, "sw.js"), "utf8");
    assert.match(
      sw,
      /const KILL_SWITCH = false;/,
      "KILL_SWITCH is on — the service worker will unregister itself for every visitor"
    );
  });

  test("sitemap lists the real pages and skips the utility ones", async () => {
    const xml = await readFile(join(SITE, "sitemap.xml"), "utf8");
    assert.ok(xml.includes("https://jazzikp.github.io/about/"), "about page missing from sitemap");
    assert.ok(xml.includes("/2026/08/09/kimi-k3-attention-residuals/"), "posts missing from sitemap");
    for (const excluded of ["/write/", "/404", "/offline", "/tags/", "/corpus.json"]) {
      assert.ok(!xml.includes(excluded), `${excluded} should not be in the sitemap`);
    }
  });

  test("robots.txt points at the sitemap", async () => {
    const txt = await readFile(join(SITE, "robots.txt"), "utf8");
    assert.match(txt, /Sitemap: https:\/\/jazzikp\.github\.io\/sitemap\.xml/);
    assert.match(txt, /Disallow: \/corpus\.json/, "corpus.json should stay out of search indexes");
  });

  test("the chat corpus is valid JSON covering posts and bio pages", async () => {
    const corpus = JSON.parse(await readFile(join(SITE, "corpus.json"), "utf8"));
    const pageUrls = new Set((corpus.pages || []).map((p) => p.url));
    for (const need of ["/", "/about/", "/contact/", "/projects/", "/blogs/", "/reports/"]) {
      assert.ok(pageUrls.has(need), `corpus pages missing ${need}`);
    }
    assert.ok(!pageUrls.has("/write/"), "the write tool should not be in the corpus");
    const posts = corpus.posts || [];
    assert.ok(posts.length >= 5, `corpus has ${posts.length} posts`);
    assert.ok(posts.some((p) => (p.url || "").includes("kimi-k3")));
    assert.ok(posts.some((p) => (p.title || "").includes("Kimi") || (p.text || "").includes("KDA")));
    for (const post of posts) {
      assert.ok(post.title && post.url && post.text, `post is missing fields: ${JSON.stringify(post).slice(0, 120)}`);
    }
    assert.ok((corpus.projects || []).length >= 1, "corpus has no projects");
    const about = (corpus.pages || []).find((p) => p.kind === "about" || p.url === "/about/");
    assert.ok(about && /Phoenix|Grok/.test(about.text), "about page text is missing from the corpus");
  });

  test("the RSS feed is valid XML with items", async () => {
    const xml = await readFile(join(SITE, "feed.xml"), "utf8");
    assert.match(xml, /^<\?xml/);
    assert.ok((xml.match(/<item>/g) || []).length >= 3, "feed has fewer items than expected");
  });

  test("the PWA manifest parses and its icons exist", async () => {
    const manifest = JSON.parse(await readFile(join(SITE, "pwa/manifest.json"), "utf8"));
    assert.ok(manifest.icons.length >= 2);
    for (const icon of manifest.icons) {
      const res = await fetch(site.origin + "/pwa/" + icon.src);
      assert.equal(res.status, 200, `manifest icon missing: ${icon.src}`);
    }
  });
});
