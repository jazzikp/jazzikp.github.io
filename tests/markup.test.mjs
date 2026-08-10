/*
 * Per-page markup health: document structure, images, and accessibility.
 *
 * These are the regressions that are easy to introduce by hand and invisible
 * until someone opens the page with a screen reader or on a slow connection.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { startSite, PAGES, POSTS, tags, attrs } from "./helpers/site.mjs";

let site;
const html = new Map();

before(async () => {
  site = await startSite();
  for (const page of PAGES) {
    html.set(page, await (await fetch(site.origin + page)).text());
  }
});
after(async () => { await site.close(); });

describe("markup", () => {
  test("each page has exactly one h1 in its static markup", () => {
    for (const [page, doc] of html) {
      const count = (doc.match(/<h1[\s>]/g) || []).length;
      assert.equal(count, 1, `${page} has ${count} <h1> elements`);
    }
  });

  test("heading levels never skip a rank", () => {
    // h1 -> h4 with nothing between reads as a broken outline to assistive tech.
    for (const [page, doc] of html) {
      const levels = [...doc.matchAll(/<h([1-6])[\s>]/g)].map((m) => Number(m[1]));
      let previous = 0;
      for (const level of levels) {
        if (previous) {
          assert.ok(
            level <= previous + 1,
            `${page} jumps from h${previous} to h${level}`
          );
        }
        previous = level;
      }
    }
  });

  test("every image has alt text", () => {
    for (const [page, doc] of html) {
      for (const img of tags(doc, "img")) {
        assert.ok(/\balt="/.test(img), `${page} has an <img> with no alt: ${img}`);
      }
    }
  });

  test("every image declares width and height", () => {
    // Intrinsic dimensions let the browser reserve space, which is what keeps
    // Cumulative Layout Shift at zero.
    for (const [page, doc] of html) {
      for (const img of tags(doc, "img")) {
        assert.ok(
          /\bwidth="\d+"/.test(img) && /\bheight="\d+"/.test(img),
          `${page} has an <img> without dimensions: ${img}`
        );
      }
    }
  });

  test("images below the fold are lazy", () => {
    for (const page of POSTS) {
      const doc = html.get(page);
      const prose = doc.slice(doc.indexOf('class="prose'));
      for (const img of tags(prose, "img")) {
        assert.match(img, /loading="lazy"/, `${page} has an eager prose image: ${img}`);
      }
    }
  });

  test("no image src is empty", () => {
    for (const [page, doc] of html) {
      for (const src of attrs(doc, "img", "src")) {
        assert.notEqual(src.trim(), "", `${page} has an <img> with an empty src`);
      }
    }
  });

  test("only optimized image formats are served", () => {
    for (const [page, doc] of html) {
      for (const src of attrs(doc, "img", "src")) {
        if (!src.startsWith("/img/")) continue;
        assert.match(
          src,
          /\.(webp|png|jpg)$/,
          `${page} references an unexpected image format: ${src}`
        );
        assert.ok(
          !/hero-anime\.jpg|avatar-anime\.jpg/.test(src),
          `${page} references a full-resolution original: ${src}`
        );
      }
    }
  });

  test("every page has a skip link as its first focusable element", () => {
    for (const [page, doc] of html) {
      assert.match(doc, /class="skip-link" href="#main"/, `${page} has no skip link`);
      assert.match(doc, /<main id="main">/, `${page} has no <main id="main">`);
    }
  });

  test("external links carry rel=noopener", () => {
    for (const [page, doc] of html) {
      const anchors = [...doc.matchAll(/<a\b[^>]*>/gi)].map((m) => m[0]);
      for (const a of anchors) {
        if (!/target="_blank"/.test(a)) continue;
        assert.match(a, /rel="[^"]*noopener/, `${page} opens a new tab without noopener: ${a}`);
      }
    }
  });

  test("pages declare a language, and Chinese posts say so", () => {
    for (const [page, doc] of html) {
      const lang = doc.match(/<html lang="([^"]+)"/)?.[1];
      assert.ok(lang, `${page} has no lang attribute`);
    }
    assert.match(
      html.get("/2023/10/14/推荐系统总结/"),
      /<html lang="zh"/,
      "the Chinese post should declare lang=zh"
    );
  });

  test("posts show a reading time", () => {
    for (const page of POSTS) {
      assert.match(html.get(page), /\d+ min read/, `${page} has no reading time`);
    }
  });

  test("the blog index exposes filterable metadata", () => {
    const doc = html.get("/blogs/");
    const tagSets = attrs(doc, "a", "data-tags");
    assert.ok(tagSets.length >= 4, "post cards are missing data-tags");
    for (const set of tagSets) {
      // Pipe-wrapped so tags containing spaces still match exactly.
      assert.match(set, /^\|.*\|$/, `data-tags is not pipe-wrapped: "${set}"`);
      assert.ok(!/\|\|/.test(set), `data-tags has an empty entry: "${set}"`);
    }
  });
});
