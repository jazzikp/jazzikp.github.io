(function () {
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

  function addBubble(who, text) {
    var div = document.createElement("div");
    div.className = "bubble " + who;
    div.textContent = text;
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
      form.requestSubmit();
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
                bot.textContent = full;
                log.scrollTop = log.scrollHeight;
              }
            } catch (err) {}
          });
        }
        if (!full) bot.textContent = "(empty response)";
        messages.push({ role: "assistant", content: full || "" });
      } else {
        var json = await res.json();
        var text =
          (((json.choices || [])[0] || {}).message || {}).content ||
          json.output_text ||
          json.text ||
          JSON.stringify(json);
        bot.textContent = text;
        messages.push({ role: "assistant", content: text });
      }
    } catch (err) {
      bot.textContent = "Something went wrong talking to Grok.";
      setStatus(String(err.message || err));
    } finally {
      busy = false;
    }
  }
})();
