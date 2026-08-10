import { RESUME } from "./resume.js";

const SYSTEM = `You are Jazzik on J'Log (jazzikp.github.io), Zhejian Peng's site.
You have his public resume below. Speak as Jazzik, a helper on this site — do not call yourself Grok. Use markdown (bold, lists).

When someone pastes a job description or asks "is this a good job for me / for him / a good fit":
- On this site, **default to Zhejian Peng** and use the resume. "For me" from a visitor still means him unless they give their own years, title, and stack.
- Only switch to the visitor if they clearly provide their own background.
- Structure the answer as: **Fit** (strong / mixed / weak), **Why it matches**, **Gaps**, **Verdict** (one or two sentences). Be honest. Do not invent employers, titles, or numbers that are not in the resume.
- Never output phone numbers, emails, addresses, salary, or immigration details. None of those are in the resume; do not guess.

Other questions: answer normally about his work, ranking, recsys, coding RL. Tennis and PADI diving are hobbies only.

RESUME:
${RESUME}`;

const TRANSLATE_SYSTEM = `You are Grok Translation. Translate the user's HTML blog into the requested language.
Rules:
- Output ONLY the translated HTML. No preamble, no markdown fences, no commentary.
- Preserve every HTML tag, attribute, href, class, id, and table structure.
- Preserve math (LaTeX in \\( \\), $$ $$, or MathJax). Do not translate identifiers like AttnRes, KDA, MLA, MoE, PreNorm.
- Keep numbers, paper ids, and code unchanged.
- Translate visible prose only.`;

const ALLOW_ORIGIN = [
  "https://jazzikp.github.io",
  "http://localhost:4000",
  "http://127.0.0.1:4000"
];

function corsHeaders(origin) {
  const allow = ALLOW_ORIGIN.includes(origin) ? origin : ALLOW_ORIGIN[0];
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
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
    if (request.method !== "POST") {
      return new Response("Not found", { status: 404, headers });
    }
    if (url.pathname === "/translate") {
      return translate(request, env, headers);
    }
    if (url.pathname !== "/chat") {
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
        model: "grok-4.5",
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

async function translate(request, env, headers) {
  if (!env.XAI_API_KEY) {
    return new Response("Missing XAI_API_KEY", { status: 500, headers });
  }
  let payload;
  try {
    payload = await request.json();
  } catch {
    return new Response("Invalid JSON", { status: 400, headers });
  }
  const target = typeof payload.target === "string" ? payload.target.trim().slice(0, 80) : "";
  const html = typeof payload.html === "string" ? payload.html : "";
  if (!target || !html) {
    return new Response("Need target and html", { status: 400, headers });
  }
  if (html.length > 120000) {
    return new Response("Post too long to translate", { status: 413, headers });
  }

  const upstream = await fetch("https://api.x.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.XAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "grok-4.5",
      stream: false,
      reasoning_effort: "low",
      temperature: 0,
      messages: [
        { role: "system", content: TRANSLATE_SYSTEM },
        { role: "user", content: "Target language: " + target + "\n\n" + html }
      ]
    })
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(text || "Upstream error", {
      status: upstream.status,
      headers: { ...headers, "Content-Type": "text/plain" }
    });
  }

  let out = "";
  try {
    const json = await upstream.json();
    out = (((json.choices || [])[0] || {}).message || {}).content || "";
  } catch {
    return new Response("Bad upstream JSON", { status: 502, headers });
  }
  out = out.replace(/^```(?:html)?\n?/i, "").replace(/\n?```$/i, "").trim();
  return new Response(out, {
    status: 200,
    headers: { ...headers, "Content-Type": "text/html; charset=utf-8" }
  });
}
