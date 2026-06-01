# CoT-ICL Lab — interactive explainer

A Distill-style, fully static website that explains the CoT-ICL Lab framework with
interactive widgets. No build step: plain HTML/CSS/ES-modules, with D3 and KaTeX
loaded from a CDN.

## Preview locally

ES modules require an HTTP origin (opening `index.html` via `file://` will fail
with CORS/module errors). Serve the folder:

```bash
python3 -m http.server -d docs 8000
# then open http://localhost:8000
```

## Deploy via GitHub Pages

This site is meant to be served from `docs/` on `main`:

1. Repo → **Settings → Pages**
2. **Build and deployment → Source: Deploy from a branch**
3. Branch: **`main`**, folder: **`/docs`** → **Save**

The site will publish at `https://<user>.github.io/cot-icl-lab/`. The `.nojekyll`
file disables Jekyll so nothing is rewritten.

## Layout

```
docs/
  index.html              # the article
  css/distill.css         # Distill-style typography + reading column + sidenotes
  css/widgets.css         # controls, SVG graph, token chips, two-track accents
  js/
    rng.js                # seeded PRNG + sampling helpers
    dag.js                # RandomDAG.generate_adj_list (shared skeleton)
    token-processor.js    # Track A: embeddings + FCN H + argmax (get_output_token)
    symbolic.js           # Track B: word transforms + ChatML framing
    recipe.js             # PowerLawRecipe.get_value
    widgets/*.js          # one mount() per interactive
    main.js               # mounts widgets, renders math
  assets/cot_icl_hero.png   # static hero snapshot (social-preview / og:image)
```

Each interactive is a faithful browser port of the corresponding Python in
`src/tokenized_cot_icl/`. A seeded generator makes every "Regenerate" reproducible;
values reproduce the **logic** of the pipeline but are not bit-identical to the
numpy/torch originals.
