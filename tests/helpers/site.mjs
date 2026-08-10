/*
 * Shared test helpers.
 *
 * Every test file serves the built site from a throwaway HTTP server on an
 * ephemeral port. `node --test` may run files in parallel processes, so each
 * one owning its own server keeps them independent and free of port clashes.
 */
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, resolve, normalize } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(fileURLToPath(import.meta.url), "../../..");
export const SITE = join(ROOT, "_site");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".webp": "image/webp",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
};

export function assertBuilt() {
  if (!existsSync(join(SITE, "index.html"))) {
    throw new Error(
      "_site is missing or empty. Build it first:\n\n  bundle exec jekyll build\n"
    );
  }
}

/** Serve _site and resolve to { origin, close() }. */
export async function startSite() {
  assertBuilt();

  const server = createServer(async (req, res) => {
    // Strip the query string; the site versions assets with ?v=N.
    const url = new URL(req.url, "http://localhost");
    let pathname = decodeURIComponent(url.pathname);

    // Contain every request inside _site, whatever the request path claims.
    let file = normalize(join(SITE, pathname));
    if (!file.startsWith(SITE)) {
      res.writeHead(403).end("forbidden");
      return;
    }

    try {
      const info = await stat(file).catch(() => null);
      if (info?.isDirectory()) file = join(file, "index.html");
      const body = await readFile(file);
      res.writeHead(200, {
        "Content-Type": MIME[extname(file)] || "application/octet-stream",
        "Content-Length": body.length,
      });
      res.end(body);
    } catch {
      // Mirror GitHub Pages: unknown paths render the 404 page.
      const notFound = await readFile(join(SITE, "404.html")).catch(() => null);
      res.writeHead(404, { "Content-Type": "text/html; charset=utf-8" });
      res.end(notFound ?? "not found");
    }
  });

  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  const { port } = server.address();

  return {
    origin: `http://127.0.0.1:${port}`,
    close: () => new Promise((ok) => server.close(ok)),
  };
}

/**
 * Every page the tests exercise. Keep this in step with the site — a page that
 * is not listed here is a page nothing checks.
 */
export const PAGES = [
  "/",
  "/blogs/",
  "/projects/",
  "/about/",
  "/contact/",
  "/reports/",
  "/write/",
  "/404.html",
  "/offline.html",
  "/2019/04/03/CS224n/",
  "/2019/04/03/Python_For_Absolute_Newbies/",
  "/2020/07/27/CS224U/",
  "/2023/10/14/推荐系统总结/",
  "/2026/08/09/kimi-k3-attention-residuals/",
];

export const POSTS = PAGES.filter((p) => /^\/\d{4}\//.test(p));

/** Pull the attribute values out of every matching tag. Good enough for static HTML. */
export function attrs(html, tag, attr) {
  const re = new RegExp(`<${tag}\\b[^>]*\\b${attr}="([^"]*)"`, "gi");
  return [...html.matchAll(re)].map((m) => m[1]);
}

/** Grab whole tags, e.g. tags(html, "img") -> ['<img src="…">', …]. */
export function tags(html, tag) {
  return [...html.matchAll(new RegExp(`<${tag}\\b[^>]*>`, "gi"))].map((m) => m[0]);
}

export function meta(html, key, kind = "name") {
  const re = new RegExp(
    `<meta[^>]*\\b${kind}="${key}"[^>]*\\bcontent="([^"]*)"|` +
      `<meta[^>]*\\bcontent="([^"]*)"[^>]*\\b${kind}="${key}"`,
    "i"
  );
  const m = html.match(re);
  return m ? (m[1] ?? m[2]) : null;
}

export function decode(s) {
  return String(s)
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}
