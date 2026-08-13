import { RESUME } from "./resume.js";

const SYSTEM = `You are Jazzik — my digital twin on J'Log. Speak in first person as me: casual, direct, a bit dry. Do not call yourself Grok. Do not say my legal name (Zhejian Peng) or any variant of it. Do not say "according to my resume", "based on my resume", "his resume", or "the owner of this site". Just talk as if you are me.

If someone pastes a job description or asks whether a role is a good fit:
- They mean me, unless they clearly give their own background.
- Answer as I would: honest, specific, first person ("I'd be strong on X; I'd need to ramp on Y").
- Do not invent employers, titles, or numbers that are not in the background below.

Never output phone numbers, emails, addresses, salary, or immigration details. Do not guess those.

Hobbies only: tennis, PADI diving. Use markdown (bold, lists).

BACKGROUND (use silently; never mention this block):
${RESUME}`;

const ALLOW_ORIGIN = [
  "https://jazzikp.github.io",
  "http://localhost:4000",
  "http://127.0.0.1:4000"
];

function corsHeaders(origin) {
  const allow = ALLOW_ORIGIN.includes(origin) ? origin : ALLOW_ORIGIN[0];
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin"
  };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const headers = corsHeaders(origin);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers });
    }
    const url = new URL(request.url);
    if (url.pathname === "/comments") {
      return comments(request, env, headers);
    }
    if (request.method !== "POST" || url.pathname !== "/chat") {
      return new Response("Not found", { status: 404, headers });
    }
    if (!env.XAI_API_KEY) {
      return new Response("Missing XAI_API_KEY", { status: 500, headers });
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return new Response("Invalid JSON", { status: 400, headers });
    }

    const incoming = Array.isArray(payload.messages) ? payload.messages.slice(-16) : [];
    const messages = [
      { role: "system", content: SYSTEM },
      ...incoming.filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    ];

    const upstream = await fetch("https://api.x.ai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.XAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "grok-4.6",
        stream: true,
        reasoning_effort: "low",
        messages
      })
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      return new Response(text || "Upstream error", {
        status: upstream.status,
        headers: { ...headers, "Content-Type": "text/plain" }
      });
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        ...headers,
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache"
      }
    });
  }
};

function json(headers, status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...headers, "Content-Type": "application/json; charset=utf-8" }
  });
}

function clean(s, max) {
  return String(s || "").replace(/\s+/g, " ").trim().slice(0, max);
}

async function hashIp(ip) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(ip || "unknown"));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

async function comments(request, env, headers) {
  if (!env.DB) return json(headers, 500, { error: "Comments DB is not bound" });
  const url = new URL(request.url);
  const slug = clean(url.searchParams.get("slug") || "", 200);

  if (request.method === "GET") {
    if (!slug) return json(headers, 400, { error: "Missing slug" });
    const { results } = await env.DB.prepare(
      "SELECT id, name, body, created_at FROM comments WHERE slug = ? ORDER BY id ASC LIMIT 200"
    ).bind(slug).all();
    return json(headers, 200, { comments: results || [] });
  }

  if (request.method !== "POST") {
    return json(headers, 405, { error: "Method not allowed" });
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return json(headers, 400, { error: "Invalid JSON" });
  }

  if (payload.website) return json(headers, 200, { ok: true });

  const postSlug = clean(payload.slug || slug, 200);
  const name = clean(payload.name, 40);
  const body = String(payload.body || "").trim().slice(0, 2000);
  if (!postSlug || !/^[a-zA-Z0-9/_.:-]+$/.test(postSlug)) {
    return json(headers, 400, { error: "Bad slug" });
  }
  if (name.length < 1 || body.length < 2) {
    return json(headers, 400, { error: "Name and comment are required" });
  }

  const ip = request.headers.get("CF-Connecting-IP") || "";
  const ipHash = await hashIp(ip);
  const hourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM comments WHERE ip_hash = ? AND created_at > ?"
  ).bind(ipHash, hourAgo).first();
  if ((recent && recent.n) >= 8) {
    return json(headers, 429, { error: "Too many comments. Try later." });
  }

  const created = new Date().toISOString();
  const info = await env.DB.prepare(
    "INSERT INTO comments (slug, name, body, created_at, ip_hash) VALUES (?, ?, ?, ?, ?)"
  ).bind(postSlug, name, body, created, ipHash).run();

  return json(headers, 201, {
    ok: true,
    comment: { id: info.meta.last_row_id, name, body, created_at: created }
  });
}
