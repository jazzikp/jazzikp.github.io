/*
 * Post-page niceties: copy buttons on code blocks and linkable headings.
 * Deferred, dependency-free, and a no-op on pages without prose.
 */
(function () {
  var prose = document.querySelector(".prose");
  if (!prose) return;

  /* ---- Copy buttons ------------------------------------------------- */

  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for http:// and older Safari.
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy") ? resolve() : reject();
      } catch (e) {
        reject(e);
      } finally {
        document.body.removeChild(ta);
      }
    });
  }

  prose.querySelectorAll("pre").forEach(function (pre) {
    if (pre.parentElement.classList.contains("code-block")) return;

    var wrap = document.createElement("div");
    wrap.className = "code-block";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-copy";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy code to clipboard");

    btn.addEventListener("click", function () {
      copy(pre.innerText)
        .then(function () { flash("Copied"); })
        .catch(function () { flash("Press ⌘C"); });
    });

    var timer;
    function flash(label) {
      btn.textContent = label;
      btn.classList.add("is-done");
      clearTimeout(timer);
      timer = setTimeout(function () {
        btn.textContent = "Copy";
        btn.classList.remove("is-done");
      }, 1600);
    }

    wrap.appendChild(btn);
  });

  /* ---- Heading anchors ---------------------------------------------- */

  var used = Object.create(null);

  function slugify(text) {
    var base =
      text
        .toLowerCase()
        .replace(/[\s]+/g, "-")
        .replace(/[^\w\u4e00-\u9fa5-]/g, "")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "") || "section";
    // Two headings can share a title; keep the ids unique.
    if (used[base] === undefined) {
      used[base] = 0;
      return base;
    }
    used[base] += 1;
    return base + "-" + used[base];
  }

  // Posts use h1 for major sections and h4 for sub-points, so anchor the whole
  // range rather than assuming an h2/h3 outline. Kramdown already emits ids for
  // most of these; slugify only fills the gaps.
  prose.querySelectorAll("h1, h2, h3, h4").forEach(function (h) {
    if (!h.id) h.id = slugify(h.textContent || "");
    else used[h.id] = 0;

    var a = document.createElement("a");
    a.className = "heading-anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });
})();
