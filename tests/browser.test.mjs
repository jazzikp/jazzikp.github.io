/*
 * Behaviour in a real browser: the things static analysis cannot see.
 *
 * Runs headless Chromium. Requests to the Grok proxy are stubbed, so the suite
 * never depends on a live Cloudflare worker or on network access.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { startSite } from "./helpers/site.mjs";

const PROXY = /workers\.dev/;

let site, browser;

before(async () => {
  site = await startSite();
  browser = await chromium.launch();
});
after(async () => {
  await browser?.close();
  await site.close();
});

/** A fresh context with the chat/comments backend stubbed out. */
async function open(path, { viewport = { width: 1280, height: 900 }, theme } = {}) {
  const context = await browser.newContext({ viewport });
  if (theme) {
    // addInitScript runs on every navigation, so only seed the preference when
    // there is none. Overwriting it each time would mask the site's own
    // persistence and make "theme survives navigation" untestable.
    await context.addInitScript(
      `if (!localStorage.getItem("theme")) localStorage.setItem("theme", ${JSON.stringify(theme)})`
    );
  }

  await context.route(PROXY, (route) => {
    const url = route.request().url();
    if (url.includes("/comments")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: JSON.stringify({
          comments: [{ name: "Ada", body: "Nice post", created_at: "2026-01-01T00:00:00Z" }],
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ choices: [{ message: { content: "Hello from the stub." } }] }),
    });
  });

  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e.message)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  const requests = [];
  page.on("request", (r) => requests.push(r.url()));

  await page.goto(site.origin + path, { waitUntil: "networkidle" });
  return { context, page, errors, requests };
}

describe("browser", () => {
  test("pages render without JavaScript errors", async () => {
    for (const path of ["/", "/blogs/", "/about/", "/projects/", "/contact/", "/2020/07/27/CS224U/"]) {
      const { context, page, errors } = await open(path);
      await page.waitForTimeout(200);
      assert.deepEqual(errors, [], `${path} logged errors`);
      await context.close();
    }
  });

  test("the theme toggle switches, persists, and repaints the meta colour", async () => {
    const { context, page } = await open("/", { theme: "light" });
    assert.equal(await page.getAttribute("html", "data-theme"), "light");

    await page.click(".theme-toggle");
    assert.equal(await page.getAttribute("html", "data-theme"), "dark");
    assert.equal(await page.getAttribute('meta[name="theme-color"]', "content"), "#0e0e10");
    assert.equal(await page.getAttribute(".theme-toggle", "aria-checked"), "true");

    // Survives navigation.
    await page.goto(site.origin + "/about/");
    assert.equal(await page.getAttribute("html", "data-theme"), "dark");
    await context.close();
  });

  test("dark mode is applied before first paint", async () => {
    // If the inline bootstrap ran late the page would flash white.
    const { context, page } = await open("/", { theme: "dark" });
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    assert.equal(bg, "rgb(14, 14, 16)");
    await context.close();
  });

  test("the chat widget loads on first click and answers", async () => {
    const { context, page, requests } = await open("/");
    assert.ok(!requests.some((u) => u.includes("chat.js")), "chat.js loaded before it was needed");

    await page.click("[data-open-chat]");
    await page.waitForSelector("#grok-chat.open", { timeout: 5000 });
    assert.ok(requests.some((u) => u.includes("chat.js")), "chat.js never loaded");

    await page.fill("#grok-input", "hi");
    await page.click(".chat-form button[type=submit]");
    assert.match(await page.locator("#grok-log .bubble.user").innerText(), /hi/);

    // A placeholder bubble appears immediately, so wait for the reply to land
    // rather than for the bubble to exist.
    await page.waitForFunction(
      () => {
        const bubbles = document.querySelectorAll("#grok-log .bubble.bot");
        const last = bubbles[bubbles.length - 1];
        return bubbles.length >= 2 && last.innerText.trim() !== "…";
      },
      null,
      { timeout: 5000 }
    );
    assert.match(await page.locator("#grok-log .bubble.bot").last().innerText(), /stub/);

    await page.click("[data-close-chat]");
    assert.ok(!(await page.locator("#grok-chat").evaluate((e) => e.classList.contains("open"))));
    await context.close();
  });

  test("blog search and tag filters narrow the list", async () => {
    const { context, page } = await open("/blogs/");
    const total = await page.locator(".post-card").count();

    await page.fill("#post-search", "stanford");
    await page.waitForTimeout(100);
    const found = await page.locator(".post-card:not([hidden])").count();
    assert.ok(found > 0 && found < total, `search matched ${found}/${total}`);

    await page.fill("#post-search", "no-such-post-exists");
    await page.waitForTimeout(100);
    assert.ok(await page.locator("#blog-empty").isVisible(), "empty state did not appear");

    await page.fill("#post-search", "");
    await page.click('#tag-filters [data-tag="stanford"]');
    await page.waitForTimeout(100);
    const tagged = await page.locator(".post-card:not([hidden])").count();
    assert.ok(tagged > 0 && tagged < total, `tag filter matched ${tagged}/${total}`);

    // Clicking the active tag clears it.
    await page.click('#tag-filters [data-tag="stanford"]');
    await page.waitForTimeout(100);
    assert.equal(await page.locator(".post-card:not([hidden])").count(), total);
    await context.close();
  });

  test("a /blogs/#Tag deep link preselects that tag", async () => {
    const { context, page } = await open("/blogs/#CS224n");
    await page.waitForTimeout(200);
    const shown = await page.locator(".post-card:not([hidden])").count();
    const total = await page.locator(".post-card").count();
    assert.ok(shown > 0 && shown < total, `deep link matched ${shown}/${total}`);
    assert.equal(await page.locator("#tag-filters .tag.is-on").innerText(), "CS224n");
    await context.close();
  });

  test("the language toggle swaps content and carries across pages", async () => {
    const { context, page } = await open("/about/");
    assert.ok(await page.locator('[data-lang-panel="en"]').first().isVisible());

    await page.click('.lang-tabs [data-lang="zh"]');
    await page.waitForTimeout(100);
    assert.ok(await page.locator('[data-lang-panel="zh"]').first().isVisible());
    assert.ok(!(await page.locator('[data-lang-panel="en"]').first().isVisible()));
    assert.equal(await page.getAttribute("html", "lang"), "zh");

    await page.goto(site.origin + "/2026/08/09/kimi-k3-attention-residuals/");
    await page.waitForTimeout(300);
    assert.match(await page.locator(".post-title").innerText(), /KDA 技术路线深读/);

    await page.click('.lang-tabs [data-lang="en"]');
    await page.waitForTimeout(100);
    assert.match(await page.locator(".post-title").innerText(), /An In-Depth Look at KDA/);
    await context.close();
  });

  test("code blocks get a working copy button", async () => {
    const { context, page } = await open("/2019/04/03/Python_For_Absolute_Newbies/");
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    const buttons = await page.locator(".code-copy").count();
    assert.ok(buttons > 0, "no copy buttons were added");

    await page.locator(".code-copy").first().click();
    await page.waitForTimeout(200);
    assert.match(await page.locator(".code-copy").first().innerText(), /Copied/);
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    assert.ok(clipboard.length > 0, "clipboard is empty after clicking Copy");
    await context.close();
  });

  test("headings get anchors that link to themselves", async () => {
    const { context, page } = await open("/2020/07/27/CS224U/");
    const count = await page.locator(".prose :is(h1,h2,h3,h4) .heading-anchor").count();
    assert.ok(count > 0, "no heading anchors");
    const consistent = await page.evaluate(() =>
      [...document.querySelectorAll(".prose .heading-anchor")].every(
        (a) => a.getAttribute("href") === "#" + a.parentElement.id && a.parentElement.id
      )
    );
    assert.ok(consistent, "an anchor does not point at its own heading");

    const ids = await page.evaluate(() =>
      [...document.querySelectorAll(".prose [id]")].map((e) => e.id)
    );
    assert.equal(new Set(ids).size, ids.length, "duplicate heading ids");
    await context.close();
  });

  test("comments load only once the section is near the viewport", async () => {
    const { context, page, requests } = await open("/2020/07/27/CS224U/");
    assert.ok(!requests.some((u) => u.includes("comments.js")), "comments.js loaded up front");

    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForSelector(".comment", { timeout: 5000 });
    assert.match(await page.locator(".comment").first().innerText(), /Ada/);
    await context.close();
  });

  test("images are never stretched", async () => {
    for (const path of ["/", "/about/", "/2020/07/27/CS224U/"]) {
      const { context, page } = await open(path);
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(500);
      const bad = await page.evaluate(() =>
        [...document.images]
          .filter((i) => i.naturalWidth && i.getBoundingClientRect().width)
          .filter((i) => {
            const r = i.getBoundingClientRect();
            if (getComputedStyle(i).objectFit === "cover") return false;
            return Math.abs(r.width / r.height - i.naturalWidth / i.naturalHeight) > 0.05;
          })
          .map((i) => i.currentSrc)
      );
      assert.deepEqual(bad, [], `${path} renders distorted images`);
      await context.close();
    }
  });

  test("the layout does not shift while loading", async () => {
    const { context, page } = await open("/");
    const cls = await page.evaluate(
      () =>
        new Promise((resolve) => {
          let total = 0;
          new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) if (!entry.hadRecentInput) total += entry.value;
          }).observe({ type: "layout-shift", buffered: true });
          setTimeout(() => resolve(total), 600);
        })
    );
    assert.ok(cls < 0.1, `cumulative layout shift is ${cls.toFixed(4)}`);
    await context.close();
  });

  test("mobile: the menu opens and nothing overflows sideways", async () => {
    for (const path of ["/", "/blogs/", "/about/", "/2020/07/27/CS224U/"]) {
      const { context, page } = await open(path, { viewport: { width: 390, height: 844 } });
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth
      );
      assert.ok(overflow <= 0, `${path} overflows by ${overflow}px on a 390px screen`);
      await context.close();
    }

    const { context, page } = await open("/", { viewport: { width: 390, height: 844 } });
    await page.click(".nav-toggle");
    assert.ok(await page.locator(".nav-links.open").isVisible(), "mobile menu did not open");
    await context.close();
  });

  test("keyboard users reach the skip link first", async () => {
    const { context, page } = await open("/");
    await page.keyboard.press("Tab");
    const focused = await page.evaluate(() => document.activeElement.className);
    assert.match(focused, /skip-link/, `first tab stop was "${focused}"`);
    await context.close();
  });

  test("the service worker installs and serves the site offline", async () => {
    const { context, page } = await open("/");
    await page.waitForFunction(
      () => navigator.serviceWorker.controller || navigator.serviceWorker.getRegistration(),
      null,
      { timeout: 5000 }
    );
    await page.waitForTimeout(1500);

    await context.setOffline(true);
    const res = await page.goto(site.origin + "/", { waitUntil: "domcontentloaded" });
    assert.ok(res, "offline navigation failed outright");
    assert.match(await page.title(), /J'Log/, "offline page did not render");
    const styled = await page.evaluate(
      () => getComputedStyle(document.body).fontFamily.includes("Source Serif")
    );
    assert.ok(styled, "offline page rendered without its stylesheet");

    await context.setOffline(false);
    await context.close();
  });

  test("the service worker never caches the Grok proxy", async () => {
    const { context, page } = await open("/");
    await page.waitForTimeout(1200);
    const cached = await page.evaluate(async () => {
      const names = await caches.keys();
      const urls = [];
      for (const n of names) {
        for (const req of await (await caches.open(n)).keys()) urls.push(req.url);
      }
      return urls;
    });
    assert.ok(
      !cached.some((u) => /workers\.dev/.test(u)),
      "a Grok proxy response was written to the cache"
    );
    await context.close();
  });
});
