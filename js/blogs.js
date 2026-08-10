/*
 * Client-side filtering for the blog index.
 *
 * Everything is already in the DOM, so filtering is a class toggle — no
 * fetching, no index to build, and it keeps working with the network off.
 * Post pages link here as /blogs/#TagName, so a hash preselects that tag.
 */
(function () {
  var search = document.getElementById("post-search");
  var filters = document.getElementById("tag-filters");
  var list = document.getElementById("post-list");
  var empty = document.getElementById("blog-empty");
  if (!list) return;

  var cards = Array.prototype.slice.call(list.querySelectorAll(".post-card"));
  var activeTag = "";

  function apply() {
    var q = (search && search.value || "").trim().toLowerCase();
    var shown = 0;

    cards.forEach(function (card) {
      // data-tags looks like |nlp|stanford|cs224n| — see blogs.html.
      var tags = card.getAttribute("data-tags") || "";
      var haystack = (card.getAttribute("data-title") || "") + " " + tags.replace(/\|/g, " ");
      var matchesTag = !activeTag || tags.indexOf("|" + activeTag + "|") !== -1;
      var matchesText = !q || haystack.indexOf(q) !== -1;
      var show = matchesTag && matchesText;
      card.hidden = !show;
      if (show) shown++;
    });

    if (empty) empty.hidden = shown !== 0;
  }

  function selectTag(tag) {
    activeTag = tag || "";
    if (filters) {
      filters.querySelectorAll("[data-tag]").forEach(function (btn) {
        btn.classList.toggle("is-on", (btn.getAttribute("data-tag") || "") === activeTag);
      });
    }
    apply();
  }

  if (search) {
    search.addEventListener("input", apply);
    // Escape clears the box, which is what people expect from a search field.
    search.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        search.value = "";
        apply();
      }
    });
  }

  if (filters) {
    filters.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-tag]");
      if (!btn) return;
      var tag = btn.getAttribute("data-tag") || "";
      selectTag(tag === activeTag ? "" : tag);
      var hash = tag ? "#" + btn.textContent.trim() : " ";
      history.replaceState(null, "", tag ? hash : location.pathname);
    });
  }

  function fromHash() {
    var tag = decodeURIComponent((location.hash || "").replace(/^#/, "")).toLowerCase();
    selectTag(tag);
  }

  window.addEventListener("hashchange", fromHash);
  if (location.hash) fromHash();
  else apply();
})();
