/*
 * Encryption for the invite-only section (/secret-life/).
 *
 * Shared by scripts/private.mjs, which writes the bundle, and the tests. The
 * browser side is js/private.js and must stay in step with this format.
 *
 *   manifest.json   { v, kdf: { name, hash, iterations, salt }, slots: [{ iv, data }], index: { iv, data } }
 *   <id>.json       { iv, data }
 *
 * One random 256-bit content key encrypts the index and every post with
 * AES-256-GCM. Each invitation code derives a key-encryption key with
 * PBKDF2-SHA-256 over the shared salt, and that key wraps the content key into
 * one slot. A reader's code is derived once and tried against every slot, so
 * unlocking costs one PBKDF2 run however many codes exist. Slots are shuffled
 * and padded with random decoys so the file does not reveal who holds which
 * code or exactly how many codes there are.
 *
 * Generated codes carry about 78 random bits. That, not PBKDF2, is what
 * protects the posts: the ciphertext is public, so a guessable code can be
 * brute-forced offline however many iterations each guess costs. The author
 * may still choose a phrase (`invite --code`); the CLI warns when it is short.
 */
import { webcrypto } from "node:crypto";

const subtle = webcrypto.subtle;
export const VERSION = 1;
export const ITERATIONS = 250000;
const SLOT_PADDING = 8;
const ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"; // no 0/O, 1/I/L, U

const b64 = (bytes) => Buffer.from(bytes).toString("base64");
const unb64 = (s) => new Uint8Array(Buffer.from(s, "base64"));
const random = (n) => webcrypto.getRandomValues(new Uint8Array(n));

/** A new invitation code: 16 characters (~78 bits) shown as XXXX-XXXX-XXXX-XXXX. */
export function newCode() {
  const chars = [];
  while (chars.length < 16) {
    const [byte] = random(1);
    // Rejection sampling keeps every character equally likely.
    if (byte < 240) chars.push(ALPHABET[byte % ALPHABET.length]);
  }
  return chars.join("").match(/.{4}/g).join("-");
}

/** What the reader typed, reduced to the canonical form codes are derived from. js/private.js mirrors this. */
export function normalizeCode(input) {
  return String(input).toUpperCase().replace(/[^A-Z0-9]/g, "");
}

async function deriveKek(code, salt, iterations) {
  const base = await subtle.importKey("raw", new TextEncoder().encode(normalizeCode(code)), "PBKDF2", false, ["deriveKey"]);
  return subtle.deriveKey(
    { name: "PBKDF2", hash: "SHA-256", salt, iterations },
    base,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function seal(key, bytes) {
  const iv = random(12);
  const data = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv }, key, bytes));
  return { iv: b64(iv), data: b64(data) };
}

async function unseal(key, box) {
  return new Uint8Array(await subtle.decrypt({ name: "AES-GCM", iv: unb64(box.iv) }, key, unb64(box.data)));
}

const aesKey = (raw) => subtle.importKey("raw", raw, "AES-GCM", false, ["encrypt", "decrypt"]);
const encodeJson = (value) => new TextEncoder().encode(JSON.stringify(value));

/**
 * Encrypt a whole bundle.
 *   codes:  invitation codes that may unlock it (at least one)
 *   index:  plaintext listing, e.g. { posts: [{ id, slug, title, … }] }
 *   posts:  { [id]: plaintext object }
 * Returns { manifest, files: { [id]: box } }.
 */
export async function encryptBundle({ codes, index, posts, iterations = ITERATIONS }) {
  if (!codes.length) throw new Error("at least one invitation code is required");
  const salt = random(16);
  const raw = random(32);
  const content = await aesKey(raw);

  const slots = [];
  for (const code of codes) slots.push(await seal(await deriveKek(code, salt, iterations), raw));
  // Decoys the same size as real slots: 32-byte key + 16-byte GCM tag.
  const padded = Math.ceil(slots.length / SLOT_PADDING) * SLOT_PADDING;
  while (slots.length < padded) slots.push({ iv: b64(random(12)), data: b64(random(48)) });
  for (let i = slots.length - 1; i > 0; i--) {
    const j = random(1)[0] % (i + 1);
    [slots[i], slots[j]] = [slots[j], slots[i]];
  }

  const files = {};
  for (const [id, post] of Object.entries(posts)) files[id] = await seal(content, encodeJson(post));
  return {
    manifest: {
      v: VERSION,
      kdf: { name: "PBKDF2", hash: "SHA-256", iterations, salt: b64(salt) },
      slots,
      index: await seal(content, encodeJson(index)),
    },
    files,
  };
}

/** Recover the content key with a code, or null if the code opens no slot. */
export async function unlock(manifest, code) {
  const kek = await deriveKek(code, unb64(manifest.kdf.salt), manifest.kdf.iterations);
  for (const slot of manifest.slots) {
    try {
      return await aesKey(await unseal(kek, slot));
    } catch {
      // Wrong slot or decoy: GCM authentication fails, try the next one.
    }
  }
  return null;
}

export async function decryptJson(key, box) {
  return JSON.parse(new TextDecoder().decode(await unseal(key, box)));
}
