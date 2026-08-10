/*
 * English / Chinese toggle.
 *
 * Used by the About page and by bilingual posts, which previously had two
 * near-identical copies of this logic. One implementation now drives both:
 *
 *   .lang-tabs [data-lang="en|zh"]   the buttons
 *   [data-lang-panel="en|zh"]        the blocks they show and hide
 *   data-title-en / data-title-zh    optional swappable heading (posts)
 *   data-sub-en   / data-sub-zh      optional swappable subtitle (posts)
 *
 * The choice is remembered, so picking 中 once carries across the site.
 */
(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  ready(function () {
    var tabs = document.querySelectorAll(".lang-tabs [data-lang]");
    if (!tabs.length) return;

    var panels = document.querySelectorAll("[data-lang-panel]");
    var title = document.querySelector(".post-title");
    var sub = document.querySelector(".post-sub");

    function setLang(lang, persist) {
      tabs.forEach(function (t) {
        var on = t.getAttribute("data-lang") === lang;
        t.classList.toggle("on", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach(function (p) {
        p.hidden = p.getAttribute("data-lang-panel") !== lang;
      });

      if (title) {
        var next = title.getAttribute(lang === "zh" ? "data-title-zh" : "data-title-en");
        if (next) {
          // Keep the trailing WIP badge intact — only the text node changes.
          var badge = title.querySelector(".wip-badge");
          title.childNodes[0].textContent = next + (badge ? " " : "");
        }
      }
      if (sub) {
        var s = sub.getAttribute(lang === "zh" ? "data-sub-zh" : "data-sub-en");
        if (s) sub.textContent = s;
      }

      // Screen readers and browser translation both key off this.
      document.documentElement.lang = lang === "zh" ? "zh" : "en";

      if (persist) {
        try { localStorage.setItem("post-lang", lang); } catch (e) {}
      }
      if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        setLang(tab.getAttribute("data-lang"), true);
      });
    });

    var saved = null;
    try { saved = localStorage.getItem("post-lang"); } catch (e) {}
    if (saved === "zh" || saved === "en") setLang(saved, false);
  });
})();
