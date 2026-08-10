---
# Front matter so Jekyll can inject the asset version below.
layout: null
---
/*
 * Service worker for J'Log.
 *
 * Strategy, deliberately conservative:
 *
 *   HTML pages      network-first. The network always wins when it is
 *                   reachable, so a deploy is visible on the next load and a
 *                   bad cache can never pin visitors to a stale page. The
 *                   cached copy is only used when the network fails.
 *   Static assets   cache-first. CSS, JS, fonts and images are versioned or
 *                   content-stable, so serving them from disk is safe and
 *                   makes repeat visits render without a single round trip.
 *   Everything else passes straight through — in particular the Grok proxy,
 *                   whose chat and comment responses must never be cached.
 *
 * To switch the worker off for people who already have it installed, set
 * KILL_SWITCH to true and deploy. The worker will unregister itself and drop
 * its caches on the next visit.
 */

const KILL_SWITCH = false;

const VERSION = "v{{ site.asset_version }}";
const PAGES = `pages-${VERSION}`;
const ASSETS = `assets-${VERSION}`;
const OFFLINE = "/offline.html";

// The shell needed to render any page without the network.
const PRECACHE = [
  OFFLINE,
  "/",
  "/css/site.css?v={{ site.asset_version }}",
  "/js/site.js?v={{ site.asset_version }}",
  "/fonts/source-serif-4-400-latin.woff2",
  "/fonts/source-serif-4-600-latin.woff2",
  "/fonts/source-sans-3-600-latin.woff2",
  "/img/avatar-128.webp",
  "/img/favicon-32.png",
];

const isAsset = (url) =>
  /^\/(css|js|fonts|img|pwa)\//.test(url.pathname) || url.pathname === "/favicon.ico";

self.addEventListener("install", (event) => {
  if (KILL_SWITCH) return;
  event.waitUntil(
    caches
      .open(ASSETS)
      // addAll rejects the whole install if any single URL 404s, so add them
      // one at a time and let stragglers fall back to the network.
      .then((cache) => Promise.all(PRECACHE.map((url) => cache.add(url).catch(() => {}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();

      if (KILL_SWITCH) {
        await Promise.all(names.map((n) => caches.delete(n)));
        await self.registration.unregister();
        const clients = await self.clients.matchAll({ type: "window" });
        clients.forEach((c) => c.navigate(c.url));
        return;
      }

      await Promise.all(
        names.filter((n) => n !== PAGES && n !== ASSETS).map((n) => caches.delete(n))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  if (KILL_SWITCH) return;

  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // Grok proxy, analytics, anything remote

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request));
    return;
  }

  if (isAsset(url)) {
    event.respondWith(cacheFirst(request));
  }
});

async function networkFirst(request) {
  const cache = await caches.open(PAGES);
  try {
    const response = await fetch(request);
    if (response && response.ok) cache.put(request, response.clone());
    return response;
  } catch (err) {
    return (
      (await cache.match(request)) ||
      (await caches.match(OFFLINE)) ||
      new Response("Offline", { status: 503, headers: { "Content-Type": "text/plain" } })
    );
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) {
    const cache = await caches.open(ASSETS);
    cache.put(request, response.clone());
  }
  return response;
}
