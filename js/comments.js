(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  ready(function () {
    var root = document.getElementById("comments");
    var list = document.getElementById("comment-list");
    var form = document.getElementById("comment-form");
    var status = document.getElementById("comment-status");
    if (!root || !list || !form) return;

    var proxy = (window.GROK_PROXY || "").replace(/\/$/, "");
    var slug = root.getAttribute("data-slug") || location.pathname;

    function setStatus(text, isErr) {
      if (!status) return;
      status.hidden = !text;
      status.textContent = text || "";
      status.classList.toggle("is-err", !!isErr);
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function fmt(iso) {
      try {
        return new Date(iso).toLocaleDateString(undefined, {
          year: "numeric", month: "short", day: "numeric"
        });
      } catch (e) {
        return iso;
      }
    }

    function render(items) {
      if (!items || !items.length) {
        list.innerHTML = "<p class=\"comment-empty\">No comments yet.</p>";
        return;
      }
      list.innerHTML = items.map(function (c) {
        return "<article class=\"comment\">" +
          "<header><strong>" + escapeHtml(c.name) + "</strong>" +
          "<time>" + escapeHtml(fmt(c.created_at)) + "</time></header>" +
          "<p>" + escapeHtml(c.body).replace(/\n/g, "<br>") + "</p>" +
          "</article>";
      }).join("");
    }

    function load() {
      if (!proxy) {
        list.textContent = "Comments are not connected.";
        return;
      }
      fetch(proxy + "/comments?slug=" + encodeURIComponent(slug))
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.json();
        })
        .then(function (data) { render(data.comments || []); })
        .catch(function () {
          list.innerHTML = "<p class=\"comment-empty\">Could not load comments.</p>";
        });
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = form.querySelector("button[type=submit]");
      var fd = new FormData(form);
      var payload = {
        slug: slug,
        name: (fd.get("name") || "").toString().trim(),
        body: (fd.get("body") || "").toString().trim(),
        website: (fd.get("website") || "").toString()
      };
      if (!payload.name || payload.body.length < 2) {
        setStatus("Name and comment are required.", true);
        return;
      }
      btn.disabled = true;
      setStatus("");
      fetch(proxy + "/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
        .then(function (res) { return res.json().then(function (j) { return { ok: res.ok, body: j }; }); })
        .then(function (out) {
          if (!out.ok) throw new Error(out.body.error || "Failed to post");
          form.reset();
          setStatus("Posted.");
          load();
        })
        .catch(function (err) {
          setStatus(String(err.message || err), true);
        })
        .then(function () { btn.disabled = false; });
    });

    load();
  });
})();
