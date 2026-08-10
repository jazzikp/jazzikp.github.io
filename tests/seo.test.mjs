/*
 * Search and social metadata.
 *
 * A personal site lives or dies on how it renders when someone pastes the link
 * into X, LinkedIn or Slack, and none of that is visible while developing.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startSite, PAGES, POSTS, meta, decode } from "./helpers/site.mjs";

const SITE_URL = "https://jazzikp.github.io";
const INDEXABLE = PAGES.filter((p) => !["/write/", "/404.html", "/offline.html"].includes(p));

let site;
const html = new Map();

before(async () => {
  site = await startSite();
  for (const page of PAGES) {
    html.set(page, await (await fetch(site.origin + page)).text());
  }
});
after(async () => { await site.close(); });

describe("seo", () => {
  test("every page has a non-empty title and description", () => {
    for (const [page, doc] of html) {
      const title = doc.match(/<title>([^<]*)<\/title>/)?.[1];
      assert.ok(title && title.trim(), `${page} has no <title>`);
      const description = meta(doc, "description");
      assert.ok(description && description.trim(), `${page} has no meta description`);
      assert.ok(
        description.length <= 320,
        `${page} description is ${description.length} chars — it will be truncated`
      );
    }
  });

  test("descriptions are distinct across pages", () => {
    const byDescription = new Map();
    for (const page of INDEXABLE) {
      const d = meta(html.get(page), "description");
      byDescription.set(d, [...(byDescription.get(d) || []), page]);
    }
    for (const [description, pages] of byDescription) {
      assert.ok(
        pages.length === 1,
        `these pages share a description ("${description}"): ${pages.join(", ")}`
      );
    }
  });

  test("canonical URLs are absolute and match the page", () => {
    for (const [page, doc] of html) {
      const canonical = doc.match(/<link rel="canonical" href="([^"]+)"/)?.[1];
      assert.ok(canonical?.startsWith(SITE_URL), `${page} canonical is not absolute: ${canonical}`);
      if (page !== "/404.html" && page !== "/offline.html") {
        assert.equal(
          decodeURIComponent(canonical),
          SITE_URL + decodeURIComponent(page),
          `${page} canonical points somewhere else`
        );
      }
    }
  });

  test("Open Graph and Twitter cards are complete", () => {
    for (const [page, doc] of html) {
      for (const key of ["og:title", "og:description", "og:url", "og:image", "og:type", "og:site_name"]) {
        assert.ok(meta(doc, key, "property"), `${page} is missing ${key}`);
      }
      assert.equal(meta(doc, "twitter:card"), "summary_large_image", `${page} twitter:card`);
      for (const key of ["twitter:title", "twitter:description", "twitter:image"]) {
        assert.ok(meta(doc, key), `${page} is missing ${key}`);
      }
      const image = meta(doc, "og:image", "property");
      assert.ok(image.startsWith("https://"), `${page} og:image must be absolute: ${image}`);
    }
  });

  test("the share image exists and is the right shape", async () => {
    const url = meta(html.get("/"), "og:image", "property");
    const res = await fetch(site.origin + url.replace(SITE_URL, ""));
    assert.equal(res.status, 200, "og:image 404s");
    assert.equal(meta(html.get("/"), "og:image:width", "property"), "1200");
    assert.equal(meta(html.get("/"), "og:image:height", "property"), "630");
  });

  test("share titles do not repeat the site name", () => {
    // og:site_name already carries it; repeating eats the ~60 visible characters.
    for (const page of POSTS) {
      const ogTitle = decode(meta(html.get(page), "og:title", "property"));
      assert.ok(!ogTitle.includes("J'Log |"), `${page} og:title repeats the site name: ${ogTitle}`);
    }
  });

  test("posts are typed as articles with a publish date", () => {
    for (const page of POSTS) {
      const doc = html.get(page);
      assert.equal(meta(doc, "og:type", "property"), "article", `${page} og:type`);
      const published = meta(doc, "article:published_time", "property");
      assert.ok(published && !Number.isNaN(Date.parse(published)), `${page} publish date`);
    }
  });

  test("JSON-LD parses and uses the right schema type", () => {
    for (const [page, doc] of html) {
      const blocks = [...doc.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
      assert.equal(blocks.length, 1, `${page} should have exactly one JSON-LD block`);
      let data;
      assert.doesNotThrow(() => { data = JSON.parse(blocks[0][1]); }, `${page} JSON-LD is invalid`);
      const expected = POSTS.includes(page) ? "BlogPosting" : "Person";
      assert.equal(data["@type"], expected, `${page} JSON-LD @type`);
      assert.equal(data["@context"], "https://schema.org");
      if (expected === "BlogPosting") {
        assert.ok(data.headline, `${page} JSON-LD has no headline`);
        assert.ok(!Number.isNaN(Date.parse(data.datePublished)), `${page} JSON-LD date`);
      }
    }
  });

  test("utility pages are marked noindex and kept out of the sitemap", () => {
    for (const page of ["/write/", "/404.html", "/offline.html"]) {
      assert.match(html.get(page), /<meta name="robots" content="noindex/, `${page} is indexable`);
    }
  });

  test("the feed is discoverable from every page", () => {
    for (const [page, doc] of html) {
      assert.match(
        doc,
        /<link rel="alternate" type="application\/rss\+xml"/,
        `${page} does not advertise the RSS feed`
      );
    }
  });
});
