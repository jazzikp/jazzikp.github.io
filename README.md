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

## Invite-only posts

`/secret-life/` holds posts that only people with an invitation code can
read. You write them in `private/`, which is git-ignored and excluded from
Jekyll. A script encrypts them, and only the ciphertext is committed, to
`secret-life/data/`. Readers type their code on the page, and the browser
decrypts the posts. The code never leaves their machine.

```bash
npm run private -- new "First months at xAI"   # creates private/posts/2026-09-26-first-months-at-xai.md
npm run private -- invite "Alice"              # prints Alice's code and republishes
npm run private -- publish                     # after writing or editing a post
npm run private -- list                        # who has which code
npm run private -- revoke "Alice"              # re-encrypts everything under a new key
git add secret-life/data && git commit -m "Update secret life"
```

Posts are Markdown with `title`, optional `subtitle`, `date`, and
`draft: true` to hold a post back. They render through kramdown exactly like
public posts. Images and maths are not supported yet.

Everything under `private/` exists only on your machine: the drafts, and
`codes.json` with each invitee's code. Back it up somewhere private. If you
lose it, generate new codes and republish.

How the encryption works:

- Each publish encrypts every post with a fresh random AES-256-GCM key.
- Each invitation code wraps that key through PBKDF2-SHA-256.
- File names are random, and decoy slots hide how many codes exist.
- Codes are generated with about 78 random bits, so they cannot be guessed.
  Hand-picked codes are not supported, because anyone can download the
  ciphertext and try passwords offline.

Two limits come with the design:

- **Anyone with a code can read everything, and copy it.** Treat the section
  as shared with every invitee, never as secret from them.
- **Revoking a code only protects what you publish afterwards.** The repo is
  public, so git history keeps older ciphertext that the revoked code can
  still open.

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
`_includes/site.scss` in cascade order. `_includes/head.html` compiles that with
Jekyll's `scssify` filter and inlines the minified result (about 6 KB gzipped)
in every page, so first paint never waits on a stylesheet request. Font URLs in
the partials must be root-relative (`/fonts/…`) because the CSS runs at every
page depth.

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
