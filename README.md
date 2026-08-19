# J'Log · Zhejian Peng

Personal site: **[jazzikp.github.io](https://jazzikp.github.io)**

Notes on recommendation systems, ads ranking, and applied machine learning. I also contributed to the Grok Coding RL model at xAI. The floating avatar is a Grok chat, with an anime portrait of me as the logo.

## Write in the browser

Open [/write/](https://jazzikp.github.io/write/). Draft markdown, preview it, then either download the file or publish with a GitHub token (Contents write on this repo only). The token is not stored on the site.

## Technical reports

Add a file in `_reports/`:

```markdown
---
title: Ranking systems in production
subtitle: Targets, data, metrics
date: 2026-09-01
---

Your note here.
```

Then delete the matching row in `_data/upcoming_reports.yml`.

## Paid consult

GitHub Pages cannot charge cards. Best setup:

1. Create a [Tally](https://tally.so) form (email, question, context) and enable **Stripe** payment on submit, or
2. Create a [Stripe Payment Link](https://dashboard.stripe.com/payment-links) / [Lemon Squeezy](https://www.lemonsqueezy.com) product.

Put the result in `_config.yml`:

```yaml
consult:
  price: "$200"
  checkout_url: "https://buy.stripe.com/..."
  # or
  tally_embed: "https://tally.so/embed/xxxx"
```

Tally is the better fit for “submit a paid question.” Lemon Squeezy is better if you want them to handle sales tax. Cal.com if you later want booked calls instead of written answers.

## Local preview

```bash
bundle install
bundle exec jekyll serve
```

Open [http://localhost:4000](http://localhost:4000).

## How the front end is put together

**Styles.** Every rule lives in a partial under `_sass/` and is imported by
`css/site.scss` in cascade order. Jekyll compiles that to a single minified
`css/site.css`, so a page needs exactly one stylesheet request. Edit the
partials, never the compiled file.

**Fonts** are self-hosted in `fonts/` rather than loaded from Google, which
keeps the critical path on one origin. To refresh or change them, edit the
`FACES` list in `scripts/fetch-fonts.mjs` and run:

```bash
npm run fonts     # rewrites _sass/_fonts.scss and downloads the woff2 files
```

Request single weights, not weight *ranges* — a range makes Google return the
variable font, which for Source Serif 4 is 119 KB against 20 KB for one cut.

**Images.** `img/src/` holds full-resolution originals and is excluded from the
build. Everything the site actually serves is derived from them:

```bash
npm run images    # resizes, converts to WebP, rebuilds icons and the social card
```

Give every `<img>` a `width` and `height` so the browser can reserve space, and
`loading="lazy"` for anything below the fold. In markdown, kramdown attribute
lists do the same job:

```markdown
![Alt text](/img/thing.webp){: loading="lazy" width="820" height="264"}
```

**Scripts.** `js/site.js` is deferred and loads everywhere; it handles the theme
and the nav. The rest is loaded only when it is needed — `chat.js` on the first
click of "Ask Jazzik", `comments.js` when the comment section nears the
viewport, `post.js` and `lang.js` only on the pages that use them.

**Maths in posts.** Write `$$…$$` and nothing else. Kramdown parses it as
maths and emits the delimiter MathJax expects, picking inline or display from
the context — `$$x$$` mid-sentence becomes inline, `$$` on its own lines becomes
a centred block. Do not write `\(x\)` directly: kramdown reads the backslash as
a Markdown escape, strips it, and MathJax is handed a plain `(x)` it will never
render. That failure is silent, so `tests/math.test.mjs` checks for it.

**Caching.** `asset_version` in `_config.yml` is appended to every CSS and JS
URL and names the service worker's caches. Bump it whenever you change a
stylesheet or a script, otherwise returning visitors keep the old file.

**Service worker.** `sw.js` is network-first for HTML — a deploy is always
visible on the next load — and cache-first for static assets. To retire it for
people who already have it installed, set `KILL_SWITCH = true` in `sw.js` and
deploy once; it will then unregister itself and drop its caches.

## Grok chat

The browser never talks to xAI with a raw key. Deploy the Cloudflare worker, then put its URL in `_config.yml` as `grok.proxy_url`.

Jekyll emits `/corpus.json` at build time — every post, the homepage bio/timeline, About, projects, reports, and contact. The worker fetches that file and retrieves the relevant excerpts into the system prompt, so a new post is available after the next GitHub Pages build with no prompt rewrite. Redeploy the worker when `workers/grok-proxy.js` (or `workers/corpus.js`) changes:

```bash
cd workers
npx wrangler secret put XAI_API_KEY
npx wrangler deploy
```

## Acknowledgements

Visual design is original. The earlier Hux Blog / Jekyll / GitHub Pages lineage is gone from this repo.

## License

MIT. See [LICENSE](LICENSE).
