# 彭哲健 · Zhejian Peng

Personal site: **[jazzikp.github.io](https://jazzikp.github.io)**

Notes on recommendation systems, ads ranking, and applied machine learning. I also contributed to the Grok Coding RL model at xAI. The floating avatar is a Grok chat, with an anime portrait of me as the logo.

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
