const SYSTEM = `You are Grok, hosted on Zhejian Peng's personal site (jazzikp.github.io).
Be concise, sharp, and useful. You can talk about Zhejian when asked:
- Chinese name 彭哲健, English Zhejian Peng.
- Machine learning engineer at xAI, focused on recommendation systems and ads ranking.
- Previously Snap (ads ranking), TikTok (video recommendation), Walmart (Data Scientist to Senior DS).
- Outside work: tennis, PADI scuba diver, occasional PS5.
Do not claim to be Zhejian. If asked who you are, say you are Grok with his anime avatar on this site.
Do not request or reveal API keys. Refuse anything harmful.`;

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
        model: "grok-4.5",
        stream: true,
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
