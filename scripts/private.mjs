#!/usr/bin/env node
/*
 * Invite-only posts: write in private/, publish ciphertext to secret-life/data/.
 *
 *   npm run private -- new "Title"        start a post in private/posts/
 *   npm run private -- invite "Label"     create a random code for someone, then republish
 *   npm run private -- invite "Label" --code "your phrase"   use your own code instead
 *   npm run private -- revoke "Label"     delete their code, then republish under a new key
 *   npm run private -- list               show labels and codes
 *   npm run private -- publish            encrypt every post in private/posts/
 *
 * private/ is git-ignored and excluded from Jekyll. Nothing readable leaves it:
 * secret-life/data/ only ever holds the output of scripts/private-crypto.mjs.
 *
 * PRIVATE_DIR and PRIVATE_OUT override both paths (used by tests and dry runs).
 */
import { readFile, writeFile, mkdir, readdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { encryptBundle, newCode, normalizeCode } from "./private-crypto.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DIR = resolve(process.env.PRIVATE_DIR || join(ROOT, "private"));
const OUT = resolve(process.env.PRIVATE_OUT || join(ROOT, "secret-life/data"));
const POSTS = join(DIR, "posts");
const CODES = join(DIR, "codes.json");

const die = (msg) => { console.error(`private: ${msg}`); process.exit(1); };

async function readCodes() {
  if (!existsSync(CODES)) return [];
  return JSON.parse(await readFile(CODES, "utf8"));
}

async function writeCodes(codes) {
  await mkdir(DIR, { recursive: true });
  await writeFile(CODES, JSON.stringify(codes, null, 2) + "\n", { mode: 0o600 });
}

// Refuse to run if the plaintext could end up in the public repo.
function assertPrivateDirIsIgnored() {
  if (process.env.PRIVATE_DIR) return;
  const rel = relative(ROOT, DIR);
  const ignored = spawnSync("git", ["check-ignore", "-q", rel + "/"], { cwd: ROOT });
  if (ignored.status !== 0) die(`${rel}/ is not git-ignored; add it to .gitignore before writing anything there`);
  const tracked = spawnSync("git", ["ls-files", "--", rel], { cwd: ROOT, encoding: "utf8" });
  if (tracked.stdout.trim()) die(`git is tracking files in ${rel}/:\n${tracked.stdout}Remove them with \`git rm --cached\` first.`);
}

const slugify = (s) =>
  s.toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "post";

async function cmdNew(title) {
  if (!title) die('usage: npm run private -- new "Title"');
  assertPrivateDirIsIgnored();
  await mkdir(POSTS, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  const file = join(POSTS, `${date}-${slugify(title)}.md`);
  if (existsSync(file)) die(`${relative(ROOT, file)} already exists`);
  await writeFile(
    file,
    `---\ntitle: ${JSON.stringify(title)}\nsubtitle: ""\ndate: ${date}\n# draft: true    # uncomment to keep it out of the next publish\n---\n\nWrite in Markdown, the same flavour as the public posts.\n`
  );
  console.log(`created ${relative(ROOT, file)}`);
}

// Codes are matched ignoring case, spaces and punctuation (see normalizeCode).
const MIN_CUSTOM = 8;
const STRONG_CUSTOM = 20;

async function cmdInvite(label, { code: custom } = {}) {
  if (!label) die('usage: npm run private -- invite "Label" [--code "your phrase"]');
  assertPrivateDirIsIgnored();
  const codes = await readCodes();
  if (codes.some((c) => c.label === label)) die(`"${label}" already has a code (see \`list\`)`);
  let code = newCode();
  if (custom !== undefined) {
    const n = normalizeCode(custom).length;
    if (n < MIN_CUSTOM) die(`a custom code needs at least ${MIN_CUSTOM} letters or digits (spaces, case and punctuation are ignored)`);
    code = custom;
    if (n < STRONG_CUSTOM) {
      console.warn(
        `warning: "${custom}" is a chosen phrase, not a random code. The encrypted files are public,\n` +
          `so anyone can download them and try guesses offline; a phrase that could appear in a\n` +
          `wordlist or be guessed from your name can be cracked. Prefer ${STRONG_CUSTOM}+ characters of\n` +
          `unrelated words, or omit --code for a random one.`
      );
    }
  }
  if (codes.some((c) => normalizeCode(c.code) === normalizeCode(code))) die("that code is already in use");
  codes.push({ label, code, created: new Date().toISOString().slice(0, 10) });
  await writeCodes(codes);
  console.log(`invitation code for ${label}: ${code}`);
  await publishIfAnything();
}

// Nothing to encrypt yet: keep the codes, publish with the first post.
async function publishIfAnything() {
  if ((await postFiles()).length || existsSync(OUT)) return cmdPublish();
  console.log("no posts yet, so nothing was published; run `publish` after writing one");
}

async function postFiles() {
  return existsSync(POSTS) ? (await readdir(POSTS)).filter((f) => f.endsWith(".md")).sort() : [];
}

async function cmdRevoke(label) {
  if (!label) die('usage: npm run private -- revoke "Label"');
  assertPrivateDirIsIgnored();
  const codes = await readCodes();
  const kept = codes.filter((c) => c.label !== label);
  if (kept.length === codes.length) die(`no code labelled "${label}"`);
  await writeCodes(kept);
  console.log(`revoked ${label}`);
  if (kept.length) await publishIfAnything();
  else {
    await rm(OUT, { recursive: true, force: true });
    console.log(`no codes left; removed ${relative(ROOT, OUT)}/`);
  }
}

async function cmdList() {
  const codes = await readCodes();
  if (!codes.length) return console.log("no invitation codes yet — `npm run private -- invite \"Name\"`");
  for (const c of codes) console.log(`${c.code}  ${c.label}  (${c.created})`);
}

// kramdown renders the Markdown so private posts match the public ones exactly.
function render(sources) {
  const ruby = `
    require "json"; require "yaml"; require "date"; require "kramdown"; require "kramdown-parser-gfm"
    out = JSON.parse($stdin.read).map do |src|
      front, body = {}, src
      if (m = src.match(/\\A---\\s*\\n(.*?)\\n---\\s*\\n?(.*)\\z/m))
        front = YAML.safe_load(m[1], permitted_classes: [Date, Time]) || {}
        body = m[2]
      end
      html = Kramdown::Document.new(body, input: "GFM", hard_wrap: false, syntax_highlighter: "rouge", auto_ids: true).to_html
      { "front" => front.transform_values { |v| v.is_a?(Date) || v.is_a?(Time) ? v.strftime("%Y-%m-%d") : v }, "html" => html, "words" => body.split.size }
    end
    print JSON.generate(out)`;
  const wrapper = join(ROOT, "scripts/with-ruby.sh");
  const [cmd, args] = existsSync(wrapper) ? [wrapper, ["bundle", "exec", "ruby", "-e", ruby]] : ["bundle", ["exec", "ruby", "-e", ruby]];
  const res = spawnSync(cmd, args, { cwd: ROOT, input: JSON.stringify(sources), encoding: "utf8", maxBuffer: 64 << 20 });
  if (res.status !== 0) die(`rendering Markdown failed:\n${res.stderr || res.error}`);
  return JSON.parse(res.stdout);
}

async function cmdPublish() {
  assertPrivateDirIsIgnored();
  const codes = await readCodes();
  if (!codes.length) die('no invitation codes yet — create one with `npm run private -- invite "Name"`');

  const names = await postFiles();
  const rendered = render(await Promise.all(names.map((f) => readFile(join(POSTS, f), "utf8"))));

  const index = { published: new Date().toISOString(), posts: [] };
  const posts = {};
  rendered.forEach(({ front, html, words }, i) => {
    if (front.draft) return;
    if (!front.title) die(`${names[i]} has no title in its front matter`);
    const date = front.date || names[i].slice(0, 10);
    const slug = slugify(names[i].replace(/\.md$/, "").replace(/^\d{4}-\d{2}-\d{2}-/, ""));
    // File names are random per publish so they say nothing about the post.
    const id = randomBytes(9).toString("base64url");
    index.posts.push({ id, slug, title: front.title, subtitle: front.subtitle || "", date, minutes: Math.max(1, Math.round(words / 230)) });
    posts[id] = { html };
  });
  index.posts.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));

  const { manifest, files } = await encryptBundle({ codes: codes.map((c) => c.code), index, posts });
  await rm(OUT, { recursive: true, force: true });
  await mkdir(OUT, { recursive: true });
  await writeFile(join(OUT, "manifest.json"), JSON.stringify(manifest) + "\n");
  for (const [id, box] of Object.entries(files)) await writeFile(join(OUT, `${id}.json`), JSON.stringify(box) + "\n");
  console.log(`published ${index.posts.length} post(s) for ${codes.length} code(s) to ${relative(ROOT, OUT) || OUT}/`);
}

const [command, ...rest] = process.argv.slice(2);
const flags = {};
const words = [];
for (let i = 0; i < rest.length; i++) {
  if (rest[i] === "--code") {
    if (i + 1 >= rest.length) die("--code needs a value");
    flags.code = rest[++i];
  } else words.push(rest[i]);
}
const arg = words.join(" ").trim();
const commands = { new: cmdNew, invite: cmdInvite, revoke: cmdRevoke, list: cmdList, publish: cmdPublish };
if (!commands[command]) die(`unknown command "${command || ""}". Use one of: ${Object.keys(commands).join(", ")}`);
await commands[command](arg, flags);
