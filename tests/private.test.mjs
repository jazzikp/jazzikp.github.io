/*
 * The invite-only section never publishes anything readable.
 *
 * Drafts and invitation codes live in the git-ignored private/ folder;
 * scripts/private.mjs writes only ciphertext to secret-life/data/. These
 * tests guard the ways that could go wrong: the folder being committed or
 * built, plaintext slipping into the data files, the page being indexed or
 * fed to the chat, and the encryption format itself.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { ROOT, SITE, assertBuilt } from "./helpers/site.mjs";
import { encryptBundle, unlock, decryptJson, newCode, normalizeCode } from "../scripts/private-crypto.mjs";

const BOX_KEYS = ["data", "iv"];
const isBase64 = (s) => typeof s === "string" && /^[A-Za-z0-9+/]+={0,2}$/.test(s);

function assertBox(box, where) {
  assert.deepEqual(Object.keys(box).sort(), BOX_KEYS, `${where} has fields beyond iv/data`);
  assert.ok(isBase64(box.iv) && isBase64(box.data), `${where} is not base64 ciphertext`);
}

async function assertCiphertextOnly(dir) {
  if (!existsSync(dir)) return;
  for (const name of await readdir(dir)) {
    assert.match(name, /^(manifest|[A-Za-z0-9_-]{12})\.json$/, `unexpected file ${name} in ${dir}`);
    const json = JSON.parse(await readFile(join(dir, name), "utf8"));
    if (name === "manifest.json") {
      assert.deepEqual(Object.keys(json).sort(), ["index", "kdf", "slots", "v"], "manifest has unexpected fields");
      assert.equal(json.kdf.name, "PBKDF2");
      assert.ok(json.kdf.iterations >= 100000, "PBKDF2 iterations were lowered");
      json.slots.forEach((s, i) => assertBox(s, `manifest slot ${i}`));
      assertBox(json.index, "manifest index");
    } else {
      assertBox(json, name);
    }
  }
}

describe("invite-only section", () => {
  test("private/ is git-ignored and nothing in it is tracked", () => {
    const ignored = spawnSync("git", ["check-ignore", "-q", "private/"], { cwd: ROOT });
    assert.equal(ignored.status, 0, "private/ is not in .gitignore");
    const tracked = spawnSync("git", ["ls-files", "--", "private"], { cwd: ROOT, encoding: "utf8" });
    assert.equal(tracked.stdout.trim(), "", `git tracks files in private/:\n${tracked.stdout}`);
  });

  test("the published data is ciphertext only", async () => {
    await assertCiphertextOnly(join(ROOT, "secret-life/data"));
    assertBuilt();
    await assertCiphertextOnly(join(SITE, "secret-life/data"));
  });

  test("the build never publishes private/", () => {
    assertBuilt();
    assert.ok(!existsSync(join(SITE, "private")), "_site/private exists — is `private` in _config.yml exclude?");
  });

  test("the page stays out of search and out of the chat corpus", async () => {
    assertBuilt();
    const sitemap = await readFile(join(SITE, "sitemap.xml"), "utf8");
    assert.ok(!sitemap.includes("secret-life"), "the sitemap lists the invite-only section");
    const corpus = await readFile(join(SITE, "corpus.json"), "utf8");
    assert.ok(!corpus.includes("secret-life"), "the chat corpus includes the invite-only section");
  });

  test("codes unlock their own bundle and nothing else", async () => {
    const [a, b] = [newCode(), newCode()];
    assert.match(a, /^[A-Z2-9]{4}(-[A-Z2-9]{4}){3}$/);
    assert.equal(normalizeCode(`  ${a} `), a, "surrounding whitespace should be ignored");

    // Hyphens never occur in base64, so these markers cannot appear by chance.
    const index = { posts: [{ id: "x", slug: "s", title: "title-marker" }] };
    const post = { html: "<p>body-marker</p>" };
    const { manifest, files } = await encryptBundle({ codes: [a, b], index, posts: { x: post }, iterations: 1000 });
    assert.equal(manifest.slots.length % 8, 0, "slots are not padded with decoys");
    const text = JSON.stringify({ manifest, files });
    assert.ok(!text.includes("title-marker") && !text.includes("body-marker"), "plaintext leaked into the bundle");

    for (const code of [a, b]) {
      const key = await unlock(manifest, code);
      assert.ok(key, "a valid code did not unlock");
      assert.deepEqual(await decryptJson(key, manifest.index), index);
      assert.deepEqual(await decryptJson(key, files.x), post);
    }
    assert.equal(await unlock(manifest, newCode()), null, "an unrelated code unlocked the bundle");
  });

  test("a code matches exactly, case included", async () => {
    const { manifest } = await encryptBundle({ codes: ["jazzikIsChad"], index: { posts: [] }, posts: {}, iterations: 1000 });
    for (const typed of ["jazzikIsChad", " jazzikIsChad ", "jazzikIsChad\n"]) {
      assert.ok(await unlock(manifest, typed), `${JSON.stringify(typed)} did not unlock`);
    }
    for (const typed of ["JAZZIKISCHAD", "jazzikischad", "jazzik is chad", "jazzik-is-chad", "JazzikIsChad", "jazzikIsChad2"]) {
      assert.equal(await unlock(manifest, typed), null, `${JSON.stringify(typed)} unlocked, but only the exact code should`);
    }
  });

  test("the pre-commit hook that blocks leaks ships with the repo", async () => {
    const hook = join(ROOT, ".githooks/pre-commit");
    assert.ok(existsSync(hook), ".githooks/pre-commit is missing");
    assert.ok((await stat(hook)).mode & 0o111, ".githooks/pre-commit is not executable");
    const src = await readFile(hook, "utf8");
    assert.match(src, /\^private\//, "the hook no longer checks private/");
    assert.match(src, /secret-life\/data/, "the hook no longer checks the published data");
  });
});
