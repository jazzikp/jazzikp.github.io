/*
 * Retrieval over the chat corpus: the worker should surface the matching
 * post without needing the full text of every article in every request.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { formatSiteContext } from "../workers/corpus.js";

const corpus = {
  pages: [
    { url: "/", title: "J'Log", kind: "home", text: "Writing on ranking. At xAI I work on Phoenix and Grok Coding RL." },
    { url: "/about/", title: "About", kind: "about", text: "Member of Technical Staff. Phoenix ranking model. Grok Coding RL." },
    { url: "/contact/", title: "Contact", kind: "contact", text: "Coaching on AI, career, ranking." },
  ],
  posts: [
    {
      url: "/2026/08/09/kimi-k3-attention-residuals/",
      title: "Kimi K3: An In-Depth Look at KDA",
      date: "2026-08-09",
      kind: "post",
      tags: ["Kimi", "KDA"],
      text: "Kimi Delta Attention (KDA) moves information along the sequence.",
    },
    {
      url: "/2026/08/16/pre-norm-vs-post-norm/",
      title: "Pre-norm vs post-norm",
      date: "2026-08-16",
      kind: "post",
      tags: ["transformers"],
      text: "Residual placement and training stability in deep transformers.",
    },
  ],
  projects: [
    { title: "Difficulty-aware looped transformer", year: "2025", kind: "project", url: "https://github.com/jazzikp/difficulty_aware_looped_transformer", text: "More loops on harder problems." },
  ],
  reports: [],
  upcoming_reports: [
    { title: "Training Grok to code", subtitle: "Notes from coding RL", eta: "Soon", kind: "upcoming_report", text: "Notes from coding RL" },
  ],
};

describe("chat corpus retrieval", () => {
  test("the catalog lists every post and page", () => {
    const ctx = formatSiteContext(corpus, "hello");
    assert.match(ctx, /Kimi K3/);
    assert.match(ctx, /Pre-norm vs post-norm/);
    assert.match(ctx, /Difficulty-aware looped transformer/);
    assert.match(ctx, /Training Grok to code/);
    assert.match(ctx, /\/about\//);
  });

  test("a question about KDA retrieves the Kimi post excerpt", () => {
    const ctx = formatSiteContext(corpus, "How does Kimi Delta Attention work?");
    assert.match(ctx, /Kimi Delta Attention \(KDA\)/);
    assert.match(ctx, /SITE EXCERPTS/);
  });

  test("bio pages are always in the excerpts", () => {
    const ctx = formatSiteContext(corpus, "How does Kimi Delta Attention work?");
    assert.match(ctx, /Phoenix ranking model/);
    assert.match(ctx, /Grok Coding RL/);
  });

  test("an empty corpus yields an empty context", () => {
    assert.equal(formatSiteContext({ pages: [], posts: [] }, "hello"), "");
    assert.equal(formatSiteContext(null, "hello"), "");
  });
});
