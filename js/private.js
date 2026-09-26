/*
 * Invite-only section: unlock with a code, then decrypt the index and posts.
 *
 * The format is defined in scripts/private-crypto.mjs; keep the two in step.
 * The unwrapped content key is kept in sessionStorage so reloading or
 * following a post link does not ask for the code again. Closing the tab, or
 * pressing Lock, forgets it.
 */
(function () {
  "use strict";

  var root = document.getElementById("inner-circle");
  if (!root || !window.crypto || !crypto.subtle) return;

  var DATA = "/inner-circle/data/";
  var SESSION_KEY = "inner-circle-key";
  var subtle = crypto.subtle;

  var lockForm = document.getElementById("ic-lock");
  var codeInput = document.getElementById("ic-code");
  var submitBtn = document.getElementById("ic-submit");
  var statusEl = document.getElementById("ic-status");
  var openEl = document.getElementById("ic-open");
  var listEl = document.getElementById("ic-list");
  var emptyEl = document.getElementById("ic-empty");
  var articleEl = document.getElementById("ic-article");
  var backLink = document.getElementById("ic-back");

  var MESSAGES = {
    working: ["Checking…", "正在验证…"],
    wrong: ["That code does not open anything. Check it and try again.", "这个邀请码无效，请检查后重试。"],
    none: ["Nothing is published here yet.", "这里还没有文章。"],
    failed: ["Could not load the posts. Check your connection and try again.", "无法加载文章，请检查网络后重试。"],
    minutes: ["min read", "分钟阅读"],
  };
  var t = function (key) { return MESSAGES[key][document.documentElement.lang === "zh" ? 1 : 0]; };

  var key = null;
  var index = null;

  function b64(s) {
    var bin = atob(s);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function toB64(bytes) {
    var bin = "";
    for (var i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }
  function normalize(code) { return String(code).toUpperCase().replace(/[^A-Z0-9]/g, ""); }

  function unseal(k, box) {
    return subtle.decrypt({ name: "AES-GCM", iv: b64(box.iv) }, k, b64(box.data)).then(function (buf) {
      return new Uint8Array(buf);
    });
  }
  function importContentKey(raw) {
    return subtle.importKey("raw", raw, "AES-GCM", false, ["decrypt"]);
  }
  function decryptJson(box) {
    return unseal(key, box).then(function (bytes) { return JSON.parse(new TextDecoder().decode(bytes)); });
  }
  function fetchJson(name) {
    return fetch(DATA + name, { cache: "no-cache" }).then(function (res) {
      if (res.status === 404) { var e = new Error("missing"); e.missing = true; throw e; }
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    });
  }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.hidden = !text;
    statusEl.classList.toggle("is-error", !!isError);
  }

  // Derive once, then try every slot: decoys and other people's slots fail
  // GCM authentication, the reader's own slot yields the raw content key.
  function unlockWith(manifest, code) {
    var kdf = manifest.kdf;
    return subtle.importKey("raw", new TextEncoder().encode(normalize(code)), "PBKDF2", false, ["deriveKey"])
      .then(function (base) {
        return subtle.deriveKey(
          { name: "PBKDF2", hash: kdf.hash, salt: b64(kdf.salt), iterations: kdf.iterations },
          base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]
        );
      })
      .then(function (kek) {
        var slots = manifest.slots.slice();
        function next() {
          if (!slots.length) return null;
          return unseal(kek, slots.shift()).then(function (raw) { return raw; }, next);
        }
        return next();
      });
  }

  function remember(raw) {
    try { sessionStorage.setItem(SESSION_KEY, toB64(raw)); } catch (e) {}
  }
  function forget() {
    try { sessionStorage.removeItem(SESSION_KEY); } catch (e) {}
  }

  function openWithRawKey(raw, manifest) {
    return importContentKey(raw).then(function (k) {
      key = k;
      return decryptJson(manifest.index);
    }).then(function (idx) {
      index = idx;
      lockForm.hidden = true;
      openEl.hidden = false;
      renderList();
      route();
    });
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function renderList() {
    listEl.textContent = "";
    emptyEl.hidden = index.posts.length > 0;
    index.posts.forEach(function (p) {
      var a = el("a", "post-card");
      a.href = "#" + p.slug;
      var time = el("time", null, formatDate(p.date, "month"));
      time.setAttribute("datetime", p.date);
      var body = el("div");
      body.appendChild(el("h2", null, p.title));
      if (p.subtitle) body.appendChild(el("p", null, p.subtitle));
      a.appendChild(time);
      a.appendChild(body);
      listEl.appendChild(a);
    });
  }

  function formatDate(iso, style) {
    var d = new Date(iso + "T00:00:00");
    if (isNaN(d)) return iso;
    var lang = document.documentElement.lang === "zh" ? "zh-CN" : "en-US";
    return d.toLocaleDateString(lang, style === "month" ? { year: "numeric", month: "short" } : { year: "numeric", month: "long", day: "numeric" });
  }

  function showList() {
    articleEl.hidden = true;
    backLink.hidden = true;
    listEl.hidden = false;
    emptyEl.hidden = index.posts.length > 0;
  }

  function showPost(p) {
    return fetchJson(p.id + ".json").then(decryptJson).then(function (post) {
      document.getElementById("ic-title").textContent = p.title;
      var sub = document.getElementById("ic-subtitle");
      sub.textContent = p.subtitle || "";
      sub.hidden = !p.subtitle;
      document.getElementById("ic-meta").textContent = formatDate(p.date) + " · " + p.minutes + " " + t("minutes");
      // The HTML comes from the author's own Markdown, rendered by kramdown and
      // authenticated by AES-GCM, so only someone holding the key could alter it.
      document.getElementById("ic-body").innerHTML = post.html;
      listEl.hidden = true;
      emptyEl.hidden = true;
      articleEl.hidden = false;
      backLink.hidden = false;
      window.scrollTo(0, root.getBoundingClientRect().top + window.scrollY - 80);
    });
  }

  function route() {
    if (!index) return;
    var slug = decodeURIComponent(location.hash.slice(1));
    var post = index.posts.filter(function (p) { return p.slug === slug; })[0];
    if (post) {
      showPost(post).catch(function () { showList(); setStatus(t("failed"), true); });
    } else {
      showList();
    }
  }

  lockForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var code = codeInput.value;
    if (normalize(code).length < 8) { setStatus(t("wrong"), true); return; }
    submitBtn.disabled = true;
    setStatus(t("working"));
    fetchJson("manifest.json")
      .then(function (manifest) {
        return unlockWith(manifest, code).then(function (raw) {
          if (!raw) { setStatus(t("wrong"), true); codeInput.select(); return; }
          remember(raw);
          codeInput.value = "";
          setStatus("");
          return openWithRawKey(raw, manifest);
        });
      })
      .catch(function (err) { setStatus(t(err && err.missing ? "none" : "failed"), true); })
      .then(function () { submitBtn.disabled = false; });
  });

  document.getElementById("ic-leave").addEventListener("click", function () {
    forget();
    key = null;
    index = null;
    listEl.textContent = "";
    document.getElementById("ic-body").textContent = "";
    openEl.hidden = true;
    lockForm.hidden = false;
    history.replaceState(null, "", location.pathname);
    codeInput.focus();
  });

  backLink.addEventListener("click", function (e) {
    e.preventDefault();
    history.pushState(null, "", location.pathname);
    showList();
  });
  window.addEventListener("hashchange", route);

  // Returning within the same tab session: reuse the key instead of the code.
  var saved = null;
  try { saved = sessionStorage.getItem(SESSION_KEY); } catch (e) {}
  if (saved) {
    fetchJson("manifest.json")
      .then(function (manifest) { return openWithRawKey(b64(saved), manifest); })
      .catch(function () { forget(); });
  }
})();
