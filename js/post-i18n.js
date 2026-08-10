(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  ready(function () {
    var tabs = document.querySelectorAll(".post-i18n [data-lang]");
    if (tabs.length) {
      var title = document.querySelector(".post-title");
      var sub = document.querySelector(".post-sub");
      var panels = document.querySelectorAll("[data-lang-panel]");
      function setLang(lang) {
        tabs.forEach(function (t) { t.classList.toggle("on", t.getAttribute("data-lang") === lang); });
        panels.forEach(function (p) {
          p.hidden = p.getAttribute("data-lang-panel") !== lang;
        });
        if (title) {
          var next = title.getAttribute(lang === "zh" ? "data-title-zh" : "data-title-en");
          if (next) {
            var badge = title.querySelector(".wip-badge");
            title.childNodes[0].textContent = next + (badge ? " " : "");
          }
        }
        if (sub) {
          var s = sub.getAttribute(lang === "zh" ? "data-sub-zh" : "data-sub-en");
          if (s) sub.textContent = s;
        }
        try { localStorage.setItem("post-lang", lang); } catch (e) {}
        if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
      }
      tabs.forEach(function (tab) {
        tab.addEventListener("click", function () { setLang(tab.getAttribute("data-lang")); });
      });
      var saved = null;
      try { saved = localStorage.getItem("post-lang"); } catch (e) {}
      if (saved === "zh" || saved === "en") setLang(saved);
      return;
    }

    var box = document.querySelector(".grok-translate");
    var article = document.querySelector(".post-container");
    if (!box || !article) return;

    var select = box.querySelector(".tr-lang");
    var custom = box.querySelector(".tr-custom");
    var go = box.querySelector(".tr-go");
    var reset = box.querySelector(".tr-reset");
    var status = box.querySelector(".tr-status");
    var busyEl = box.querySelector(".tr-busy");
    var original = article.innerHTML;
    var proxy = (window.GROK_PROXY || "").replace(/\/$/, "");
    var busy = false;

    function setBusy(on) {
      box.classList.toggle("is-busy", on);
      if (busyEl) busyEl.hidden = !on;
      go.disabled = on;
    }

    function setStatus(text, isErr) {
      if (!status) return;
      status.hidden = !text;
      status.textContent = text || "";
      status.classList.toggle("is-err", !!isErr);
    }

    function targetLang() {
      return (custom.value || "").trim() || (select && select.value) || "";
    }

    reset.addEventListener("click", function () {
      article.innerHTML = original;
      reset.hidden = true;
      setStatus("");
    });

    go.addEventListener("click", function () {
      var lang = targetLang();
      if (!lang || busy) return;
      if (!proxy) {
        setStatus("Translation is not connected.", true);
        return;
      }
      busy = true;
      setStatus("");
      setBusy(true);
      fetch(proxy + "/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html: original, target: lang })
      })
        .then(function (res) {
          if (!res.ok) return res.text().then(function (t) { throw new Error(t || ("HTTP " + res.status)); });
          return res.text().then(function (html) {
            return { html: html, cached: (res.headers.get("X-Translate-Cache") || "") === "hit" };
          });
        })
        .then(function (data) {
          if (!data.html || data.html.length < 20) throw new Error("Empty translation");
          article.innerHTML = data.html;
          reset.hidden = false;
          setStatus(data.cached ? "From cache · " + lang : "Translated into " + lang + ".");
          if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        })
        .catch(function (err) {
          setStatus(String(err.message || err), true);
        })
        .then(function () {
          busy = false;
          setBusy(false);
        });
    });
  });
})();
