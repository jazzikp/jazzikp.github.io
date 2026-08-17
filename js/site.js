(function () {
  var toggle = document.querySelector(".nav-toggle");
  var links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      links.classList.toggle("open");
    });
  }

  var root = document.documentElement;
  var btn = document.querySelector(".theme-toggle");
  var meta = document.querySelector('meta[name="theme-color"]');
  var KEY = "theme";

  function systemDark() {
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function current() {
    return root.getAttribute("data-theme") || (systemDark() ? "dark" : "light");
  }
  function apply(theme, persist) {
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;
    if (meta) meta.setAttribute("content", theme === "dark" ? "#282c34" : "#f7f5f0");
    if (btn) {
      var dark = theme === "dark";
      btn.setAttribute("aria-checked", dark ? "true" : "false");
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    }
    if (persist) {
      try { localStorage.setItem(KEY, theme); } catch (e) {}
    }
  }

  apply(current(), false);

  if (btn) {
    btn.addEventListener("click", function () {
      apply(current() === "dark" ? "light" : "dark", true);
    });
  }

  try {
    if (!localStorage.getItem(KEY) && window.matchMedia) {
      matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
        if (!localStorage.getItem(KEY)) apply(e.matches ? "dark" : "light", false);
      });
    }
  } catch (e) {}

  /*
   * English / Chinese toggle.
   *
   * The toggle sits in the navigation, so it is on every page and belongs in
   * the script that is already on every page. Two kinds of translated content
   * react to it:
   *
   *   [data-lang-panel="en|zh"]        whole blocks, shown one at a time —
   *                                    post bodies and long page prose
   *   [data-i18n-en] / [data-i18n-zh]  one element's text, swapped in place —
   *                                    nav links, buttons, headings
   *
   * Swapping text rather than shipping two copies of an element keeps one
   * <h1> per page and one link per destination, which is what the markup and
   * SEO tests require.
   */
  var tabs = document.querySelectorAll(".lang-tabs [data-lang]");
  if (!tabs.length) return;

  var panels = document.querySelectorAll("[data-lang-panel]");
  var swappable = document.querySelectorAll("[data-i18n-en]");
  var LANG_KEY = "post-lang";

  function swapText(el, lang) {
    var next = el.getAttribute(lang === "zh" ? "data-i18n-zh" : "data-i18n-en");
    if (!next) return;
    // Only the leading text node is replaced, so a trailing badge or icon —
    // the WIP marker on a post title, the mark in the footer — survives.
    var first = el.childNodes[0];
    if (first && first.nodeType === 3) {
      first.textContent = next + (el.children.length ? " " : "");
    } else {
      el.textContent = next;
    }
  }

  function setLang(lang, persist) {
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-lang") === lang;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.forEach(function (p) {
      p.hidden = p.getAttribute("data-lang-panel") !== lang;
    });
    swappable.forEach(function (el) {
      swapText(el, lang);
    });

    // Screen readers and browser translation both key off this.
    root.lang = lang === "zh" ? "zh" : "en";

    if (persist) {
      try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
    }
    if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      setLang(tab.getAttribute("data-lang"), true);
    });
  });

  var savedLang = null;
  try { savedLang = localStorage.getItem(LANG_KEY); } catch (e) {}
  if (savedLang === "zh" || savedLang === "en") setLang(savedLang, false);
})();
