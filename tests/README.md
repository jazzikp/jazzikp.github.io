# Tests

Everything here runs against `_site`, the built output — the same bytes GitHub
Pages serves. Build first, then test:

```bash
bundle exec jekyll build
cd tests
npm install
npx playwright install chromium   # first time only
npm test
```

`npm run test:fast` skips the browser suite (runs in well under a second) and
`npm run test:browser` runs only that suite.

## What each file covers

| File | Checks |
| --- | --- |
| `build.test.mjs` | Every page returns 200; generated files exist (including `/corpus.json`); the stylesheet is compiled and minified; `_sass/` and `img/src/` stay unpublished; the service worker's cache version matches `asset_version` and everything it precaches is real; sitemap, robots, feed and manifest are well formed; the chat corpus JSON covers posts and bio pages. |
| `links.test.mjs` | Crawls the site from `/` and resolves every internal link, `src` and `srcset` candidate. Also checks the 404 page and the `/tags/` redirect. |
| `markup.test.mjs` | One `<h1>` per page, no skipped heading ranks, alt text and intrinsic dimensions on every image, lazy loading in prose, `rel="noopener"` on new-tab links, a skip link on every page, and a correct `lang`. |
| `seo.test.mjs` | Titles and descriptions present, distinct and not over-long; canonical URLs absolute and self-consistent; complete Open Graph and Twitter cards; JSON-LD that parses and uses the right schema type; utility pages marked `noindex`. |
| `perf.test.mjs` | Byte budgets from `budgets.json`: critical path, stylesheet, first-load JavaScript, per-asset ceilings, total site size. Also asserts no render-blocking or third-party resources and that fonts stay self-hosted. |
| `math.test.mjs` | Posts use `$$…$$` rather than raw `\(…\)`, which kramdown silently strips; no unrendered LaTeX survives into the prose; MathJax is loaded (and async) wherever maths appears; and the number of expressions written matches the number that reach the page. |
| `secrets.test.mjs` | `_config.yml` and the rest of the source tree do not assign a Gitalk-style OAuth `clientSecret` / `clientID`. The published site must not ship those credentials again. |
| `chat-corpus.test.mjs` | The worker's retrieval over `/corpus.json` lists every page and post in the catalog and pulls the matching excerpt (KDA, bio) into the system prompt. |
| `browser.test.mjs` | Real Chromium: no console errors, theme toggle and persistence, lazy chat, blog search and tag filters, language toggle, copy buttons, heading anchors, lazy comments, image distortion, layout shift, mobile overflow, keyboard focus order, and offline rendering through the service worker. |

The chat and comments backend is stubbed with Playwright routing, so the suite
never touches the network or depends on the Cloudflare worker being up.

## Performance budgets

`budgets.json` holds the byte limits, with a note on each explaining what it
covers. When a change pushes past one, the first question is whether the change
can be made smaller. If the budget genuinely needs to move, raise it in the same
pull request and say why.

## Adding a page

Add its path to `PAGES` in `helpers/site.mjs`. Every markup, SEO and budget test
iterates that list, so one line puts a new page under the full set of checks.
