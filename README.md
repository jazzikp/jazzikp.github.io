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

## Grok chat

The browser never talks to xAI with a raw key. Deploy the Cloudflare worker, then put its URL in `_config.yml` as `grok.proxy_url`.

```bash
cd workers
npx wrangler secret put XAI_API_KEY
npx wrangler deploy
```

## Acknowledgements

Visual design is original. The earlier Hux Blog / Jekyll / GitHub Pages lineage is gone from this repo.

## License

MIT. See [LICENSE](LICENSE).
