/*
 * Credentials do not belong in this repository.
 *
 * Gitalk asked for a GitHub OAuth clientID and clientSecret in
 * `_config.yml` and then printed both into every post. The secret
 * landed in the initial import and was deleted in 37cb802, but git
 * still remembers it. This test is the guard against putting another
 * one in the tree — a revert, a new comment widget, a copied theme
 * snippet — so the published site cannot ship an OAuth secret again.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { extname, join, relative } from "node:path";
import { ROOT } from "./helpers/site.mjs";

const SKIP_DIRS = new Set([".git", "node_modules", "_site", "vendor"]);
const TEXT = new Set([
  ".yml",
  ".yaml",
  ".html",
  ".js",
  ".mjs",
  ".json",
  ".md",
  ".toml",
  ".scss",
  ".css",
  ".txt",
  ".liquid",
]);

// A YAML (or JS) assignment of a real value, not a comment or an empty key.
const ASSIGNED_SECRET = /^\s*clientSecret\s*[:=]\s*\S+/m;
const ASSIGNED_CLIENT_ID = /^\s*clientID\s*[:=]\s*\S+/m;
const GITALK_BLOCK = /^\s*gitalk\s*:/m;
const LIQUID_SECRET = /site\.gitalk\.(clientSecret|clientID)/;

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(entry.name)) continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* walk(path);
      continue;
    }
    if (TEXT.has(extname(entry.name))) yield path;
  }
}

describe("secrets stay out of the tree", () => {
  test("_config.yml has no OAuth client credentials", async () => {
    const config = await readFile(join(ROOT, "_config.yml"), "utf8");
    assert.doesNotMatch(
      config,
      GITALK_BLOCK,
      "_config.yml still has a gitalk: block — comments no longer use Gitalk"
    );
    assert.doesNotMatch(
      config,
      ASSIGNED_SECRET,
      "_config.yml assigns clientSecret; OAuth secrets cannot live in this repo"
    );
    assert.doesNotMatch(
      config,
      ASSIGNED_CLIENT_ID,
      "_config.yml assigns clientID; OAuth app credentials cannot live in this repo"
    );
  });

  test("no source file assigns or interpolates a client secret", async () => {
    const hits = [];
    for await (const path of walk(ROOT)) {
      const rel = relative(ROOT, path);
      // This file is allowed to mention the forbidden keys; it is the test.
      if (rel === "tests/secrets.test.mjs") continue;
      const body = await readFile(path, "utf8");
      if (ASSIGNED_SECRET.test(body) || LIQUID_SECRET.test(body)) {
        hits.push(rel);
      }
    }
    assert.deepEqual(
      hits,
      [],
      "clientSecret is back in the tree:\n  " + hits.join("\n  ")
    );
  });
});
