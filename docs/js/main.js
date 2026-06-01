// Mount widgets into their figure containers, render math, finalize layout.

import { mountHeroAnimation } from "./widgets/hero-animation.js";
import { mountConstructAbstract } from "./widgets/construct-abstract.js";
import { mountConstructSymbolic } from "./widgets/construct-symbolic.js";
import { mountHybridPrompt } from "./widgets/hybrid-prompt.js";
import { mountRecipeScheduler } from "./widgets/recipe-scheduler.js";

const WIDGETS = {
  "w-hero": mountHeroAnimation,
  "w-abstract": mountConstructAbstract,
  "w-symbolic": mountConstructSymbolic,
  "w-hybrid": mountHybridPrompt,
  "w-recipe": mountRecipeScheduler,
};

function mountAll() {
  for (const [id, mount] of Object.entries(WIDGETS)) {
    const el = document.getElementById(id);
    if (!el) continue;
    try {
      mount(el);
    } catch (err) {
      console.error(`Failed to mount widget #${id}:`, err);
      el.innerHTML = `<div class="widget"><p class="muted-note">This interactive failed to load. See the console for details.</p></div>`;
    }
  }
}

function renderMath() {
  if (typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

// Number sidenotes (aside elements) sequentially for a Distill-like feel.
function numberSidenotes() {
  document.querySelectorAll(".d-article > aside").forEach((aside, i) => {
    if (aside.querySelector(".sidenote-num")) return;
    const num = document.createElement("span");
    num.className = "sidenote-num";
    num.textContent = `${i + 1}. `;
    aside.prepend(num);
  });
}

function init() {
  renderMath();
  numberSidenotes();
  mountAll();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
