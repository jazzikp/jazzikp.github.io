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
  function setGiscusTheme(theme) {
    var frame = document.querySelector("iframe.giscus-frame");
    if (!frame || !frame.contentWindow) return;
    frame.contentWindow.postMessage(
      { giscus: { setConfig: { theme: theme === "dark" ? "noborder_dark" : "noborder_light" } } },
      "https://giscus.app"
    );
  }

  function apply(theme, persist) {
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;
    if (meta) meta.setAttribute("content", theme === "dark" ? "#080a0d" : "#f7f5f0");
    if (btn) {
      var dark = theme === "dark";
      btn.setAttribute("aria-checked", dark ? "true" : "false");
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    }
    setGiscusTheme(theme);
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
})();
