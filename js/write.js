(function () {
  var cfg = window.WRITE || {};
  var typeEl = document.getElementById("w-type");
  var dateEl = document.getElementById("w-date");
  var titleEl = document.getElementById("w-title");
  var subEl = document.getElementById("w-sub");
  var tagsEl = document.getElementById("w-tags");
  var tagsWrap = document.getElementById("w-tags-wrap");
  var bodyEl = document.getElementById("w-body");
  var preview = document.getElementById("w-preview");
  var tokenEl = document.getElementById("w-token");
  var statusEl = document.getElementById("w-status");
  if (!titleEl || !bodyEl) return;

  function today() {
    var d = new Date();
    var m = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return d.getFullYear() + "-" + m + "-" + day;
  }
  dateEl.value = today();

  function slugify(s) {
    return String(s || "untitled")
      .toLowerCase()
      .replace(/['’]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "untitled";
  }

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
    text = text.replace(/^# (.+)$/gm, "\n<h2>$1</h2>\n");
    text = text.replace(/(?:^|\n)((?:[-*] .+\n?)+)/g, function (_, block) {
      var items = block.trim().split("\n").map(function (line) {
        return "<li>" + line.replace(/^[-*] /, "") + "</li>";
      }).join("");
      return "\n<ul>" + items + "</ul>\n";
    });
    text = text.split(/\n{2,}/).map(function (part) {
      part = part.trim();
      if (!part) return "";
      if (/^<(ul|ol|h[234]|pre)/.test(part)) return part;
      return "<p>" + part.replace(/\n/g, "<br>") + "</p>";
    }).join("");
    return text.replace(/\u0000F(\d+)\u0000/g, function (_, i) { return fences[Number(i)]; }) || "";
  }

  function buildMarkdown() {
    var kind = typeEl.value;
    var date = dateEl.value || today();
    var title = (titleEl.value || "").trim() || "Untitled";
    var sub = (subEl.value || "").trim();
    var tags = (tagsEl.value || "").split(",").map(function (t) { return t.trim(); }).filter(Boolean);
    var lines = ["---"];
    if (kind === "note") lines.push("layout:     post");
    lines.push("title:      " + yamlQuote(title));
    if (sub) lines.push("subtitle:   " + yamlQuote(sub));
    lines.push("date:       " + date);
    lines.push("author:     " + (cfg.author || "Zhejian Peng"));
    if (kind === "note") {
      lines.push("catalog:    true");
      lines.push("tags:");
      if (tags.length) tags.forEach(function (t) { lines.push("    - " + t); });
      else lines.push("    - Note");
    }
    lines.push("---", "", bodyEl.value.replace(/\s+$/, ""), "");
    return lines.join("\n");
  }

  function yamlQuote(s) {
    if (/[:#{}[\],&*?|<>=!%@`]/.test(s) || s !== s.trim()) return JSON.stringify(s);
    return s;
  }

  function pathFor() {
    var date = dateEl.value || today();
    var slug = slugify(titleEl.value);
    if (typeEl.value === "note") return "_posts/" + date + "-" + slug + ".md";
    return "_reports/" + slug + ".md";
  }

  function setStatus(msg, ok) {
    statusEl.textContent = msg || "";
    statusEl.className = "write-status" + (ok === false ? " bad" : ok ? " ok" : "");
  }

  function refreshPreview() {
    var title = (titleEl.value || "Untitled").trim();
    var sub = (subEl.value || "").trim();
    preview.innerHTML = "<h1>" + escapeHtml(title) + "</h1>" +
      (sub ? "<p><em>" + escapeHtml(sub) + "</em></p>" : "") +
      renderMarkdown(bodyEl.value);
  }

  function toggleTags() {
    tagsWrap.style.display = typeEl.value === "note" ? "" : "none";
  }

  ["input", "change"].forEach(function (ev) {
    titleEl.addEventListener(ev, refreshPreview);
    subEl.addEventListener(ev, refreshPreview);
    bodyEl.addEventListener(ev, refreshPreview);
    typeEl.addEventListener(ev, toggleTags);
  });
  toggleTags();
  refreshPreview();

  document.getElementById("w-download").addEventListener("click", function () {
    var blob = new Blob([buildMarkdown()], { type: "text/markdown" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = pathFor().split("/").pop();
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus("Downloaded " + pathFor(), true);
  });

  document.getElementById("w-publish").addEventListener("click", async function () {
    var token = (tokenEl.value || "").trim();
    if (!token) {
      setStatus("Paste a GitHub token with write access to this repo.", false);
      return;
    }
    if (!(titleEl.value || "").trim()) {
      setStatus("Add a title first.", false);
      return;
    }
    var path = pathFor();
    var content = buildMarkdown();
    var api = "https://api.github.com/repos/" + cfg.owner + "/" + cfg.repo + "/contents/" + path;
    setStatus("Publishing " + path + "…");
    try {
      var sha = null;
      var getRes = await fetch(api + "?ref=" + encodeURIComponent(cfg.branch || "main"), {
        headers: { Authorization: "Bearer " + token, Accept: "application/vnd.github+json" }
      });
      if (getRes.ok) {
        var existing = await getRes.json();
        sha = existing.sha;
      } else if (getRes.status !== 404) {
        throw new Error(await getRes.text());
      }
      var body = {
        message: (sha ? "Update " : "Add ") + path,
        content: btoa(unescape(encodeURIComponent(content))),
        branch: cfg.branch || "main"
      };
      if (sha) body.sha = sha;
      var putRes = await fetch(api, {
        method: "PUT",
        headers: {
          Authorization: "Bearer " + token,
          Accept: "application/vnd.github+json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify(body)
      });
      var out = await putRes.json();
      if (!putRes.ok) throw new Error(out.message || JSON.stringify(out));
      var siteUrl = typeEl.value === "note"
        ? "/" + (dateEl.value || today()).replace(/-/g, "/") + "/" + slugify(titleEl.value) + "/"
        : "/reports/" + slugify(titleEl.value) + "/";
      setStatus("Published. GitHub Pages will rebuild in about a minute: " + siteUrl, true);
    } catch (err) {
      setStatus(String(err.message || err), false);
    }
  });
})();
