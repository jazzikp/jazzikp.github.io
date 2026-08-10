(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  ready(function () {
    var tabs = document.querySelectorAll(".post-i18n [data-lang]");
    if (!tabs.length) return;

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
  });
})();
