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
  var langBtn = document.querySelector(".lang-toggle");
  if (!langBtn) return;

  var panels = document.querySelectorAll("[data-lang-panel]");
  var swappable = document.querySelectorAll("[data-i18n-zh]");
  var LANG_KEY = "post-lang";
  var currentLang = "en";

  // The English is already the element's text, so it is read out of the DOM
  // rather than shipped again in an attribute. Captured before any swap runs.
  swappable.forEach(function (el) {
    var first = el.childNodes[0];
    el.__en = (first && first.nodeType === 3 ? first.textContent : el.textContent).trim();
  });

  function swapText(el, lang) {
    var next = lang === "zh" ? el.getAttribute("data-i18n-zh") : el.__en;
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
    currentLang = lang === "zh" ? "zh" : "en";
    // Label is the language you will switch TO, not the current one.
    langBtn.textContent = currentLang === "zh" ? "EN" : "中";
    langBtn.setAttribute("aria-label", currentLang === "zh" ? "Switch to English" : "Switch to Chinese");
    langBtn.setAttribute("aria-pressed", currentLang === "zh" ? "true" : "false");

    panels.forEach(function (p) {
      p.hidden = p.getAttribute("data-lang-panel") !== lang;
    });
    swappable.forEach(function (el) {
      swapText(el, lang);
    });

    // Screen readers and browser translation both key off this.
    root.lang = currentLang;

    if (persist) {
      try { localStorage.setItem(LANG_KEY, currentLang); } catch (e) {}
    }
    if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
  }

  langBtn.addEventListener("click", function () {
    setLang(currentLang === "zh" ? "en" : "zh", true);
  });

  var savedLang = null;
  try { savedLang = localStorage.getItem(LANG_KEY); } catch (e) {}
  if (savedLang === "zh" || savedLang === "en") setLang(savedLang, false);
})();
