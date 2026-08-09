(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  ready(function () {
  var panel = document.getElementById("grok-chat");
  var log = document.getElementById("grok-log");
  var form = document.getElementById("grok-form");
  var input = document.getElementById("grok-input");
  var statusEl = document.getElementById("grok-status");
  if (!panel || !form) return;

  var proxy = (window.GROK_PROXY || "").replace(/\/$/, "");
  var messages = [];
  var busy = false;

  function openChat() {
    panel.classList.add("open");
    panel.removeAttribute("hidden");
    if (!log.childElementCount) {
      addBubble("bot", "Hey — I'm Grok. Ask about Zhejian's work, ranking systems, or anything else.");
    }
    input.focus();
  }

  function closeChat() {
    panel.classList.remove("open");
  }

  document.querySelectorAll("[data-open-chat]").forEach(function (el) {
    el.addEventListener("click", function (e) {
      e.preventDefault();
      if (panel.classList.contains("open")) closeChat();
      else openChat();
    });
  });
  document.querySelectorAll("[data-close-chat]").forEach(function (el) {
    el.addEventListener("click", closeChat);
  });

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(src) {
    var text = escapeHtml(src || "").replace(/\r\n/g, "\n");
    var fences = [];
    text = text.replace(/```[\w]*\n([\s\S]*?)```/g, function (_, code) {
      fences.push("<pre><code>" + code.replace(/\n$/, "") + "</code></pre>");
      return "\u0000F" + (fences.length - 1) + "\u0000";
    });
    text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/(^|[^\*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    text = text.replace(/^### (.+)$/gm, "\n<h4>$1</h4>\n");
    text = text.replace(/^## (.+)$/gm, "\n<h3>$1</h3>\n");
    text = text.replace(/^# (.+)$/gm, "\n<h3>$1</h3>\n");
    text = text.replace(/(?:^|\n)((?:[-*] .+\n?)+)/g, function (_, block) {
      var items = block.trim().split("\n").map(function (line) {
        return "<li>" + line.replace(/^[-*] /, "") + "</li>";
      }).join("");
      return "\n<ul>" + items + "</ul>\n";
    });
    text = text.replace(/(?:^|\n)((?:\d+\. .+\n?)+)/g, function (_, block) {
      var items = block.trim().split("\n").map(function (line) {
        return "<li>" + line.replace(/^\d+\. /, "") + "</li>";
      }).join("");
      return "\n<ol>" + items + "</ol>\n";
    });
    text = text.split(/\n{2,}/).map(function (part) {
      part = part.trim();
      if (!part) return "";
      if (/^<(ul|ol|h[34]|pre)/.test(part)) return part;
      return "<p>" + part.replace(/\n/g, "<br>") + "</p>";
    }).join("");
    text = text.replace(/\u0000F(\d+)\u0000/g, function (_, i) { return fences[Number(i)]; });
    return text || "<p></p>";
  }

  function setBubble(el, who, text) {
    if (who === "bot") el.innerHTML = renderMarkdown(text);
    else el.textContent = text;
    log.scrollTop = log.scrollHeight;
  }

  function addBubble(who, text) {
    var div = document.createElement("div");
    div.className = "bubble " + who;
    setBubble(div, who, text);
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function setStatus(text) {
    if (!statusEl) return;
    statusEl.style.display = text ? "block" : "none";
    statusEl.textContent = text || "";
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy) return;
    var text = (input.value || "").trim();
    if (!text) return;
    input.value = "";
    addBubble("user", text);
    messages.push({ role: "user", content: text });
    ask(text);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (typeof form.requestSubmit === "function") form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
  });

  async function ask() {
    if (!proxy) {
      setStatus("Grok is wired up, but the server proxy is not deployed yet. That keeps the API key off this public GitHub Pages site.");
      return;
    }
    busy = true;
    setStatus("");
    var bot = addBubble("bot", "…");
    try {
      var res = await fetch(proxy + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: messages.slice(-16) })
      });
      if (!res.ok) {
        var errText = await res.text();
        throw new Error(errText || ("HTTP " + res.status));
      }
      var ctype = res.headers.get("content-type") || "";
      if (ctype.indexOf("text/event-stream") !== -1 && res.body) {
        bot.textContent = "";
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buf = "";
        var full = "";
        while (true) {
          var chunk = await reader.read();
          if (chunk.done) break;
          buf += decoder.decode(chunk.value, { stream: true });
          var parts = buf.split("\n");
          buf = parts.pop();
          parts.forEach(function (line) {
            line = line.trim();
            if (line.indexOf("data:") !== 0) return;
            var data = line.slice(5).trim();
            if (data === "[DONE]") return;
            try {
              var json = JSON.parse(data);
              var delta = ((json.choices || [])[0] || {}).delta || {};
              var piece = delta.content || "";
              if (piece) {
                full += piece;
                setBubble(bot, "bot", full);
              }
            } catch (err) {}
          });
        }
        if (!full) setBubble(bot, "bot", "(empty response)");
        messages.push({ role: "assistant", content: full || "" });
      } else {
        var json = await res.json();
        var text =
          (((json.choices || [])[0] || {}).message || {}).content ||
          json.output_text ||
          json.text ||
          JSON.stringify(json);
        setBubble(bot, "bot", text);
        messages.push({ role: "assistant", content: text });
      }
    } catch (err) {
      setBubble(bot, "bot", "Something went wrong talking to Grok.");
      setStatus(String(err.message || err));
    } finally {
      busy = false;
    }
  }
  });
})();
