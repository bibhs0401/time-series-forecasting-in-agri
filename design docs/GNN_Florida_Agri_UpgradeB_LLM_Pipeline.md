# Upgrade B in Practice: Multi-View Nodes, LLM/NLP Integration, and a Publishable Pipeline

**Companion to:** `GNN_Florida_Agri_Framework.docx`, `GNN_Florida_Agri_Robustness_Upgrade.docx`, `Trends_Pull_Plan_FL_weekly.md`
**Target:** International Journal of Forecasting / ISF
**Scope:** Concrete, build-it-this-week design for graph-based forecasting of Google Trends agricultural search interest in Florida, with realistic LLM/NLP components given a *manual* Trends-download workflow.
**Date:** June 2026

---

## 0. What this document assumes about your actual setup

These choices are grounded in the data you already have, not the abstract plan:

- **Raw data (as of this refresh):** Manual downloads in `data/g1 … g12` cover **49 search terms = 48 crops + the Cucumber anchor**, across **12** anchored comparison groups. **Cucumber is the primary anchor** (present in *every* group). Per-group contents:
  - g1: Strawberry, Blueberry, Watermelon, Cucumber*, Tomato
  - g2: Cucumber*, Potato, Lime, Bell pepper, Cabbage
  - g3: Cucumber*, Tomato, Guava, Pineapple, Papaya
  - g4: Cucumber*, Peanut, Avocado, Pumpkin, Cabbage
  - g5: Cucumber*, Lemon, Lime, Grape, Peach
  - g6: Cucumber*, Celery, Lettuce, Maize, Carrot
  - g7: Cucumber*, Grapefruit, Citrus fruit, Sweet potato, Sugarcane
  - g8: Cucumber*, Okra, Zucchini, Eggplant, Radish
  - g9: Cucumber*, Blackberry, Carambola, Citrus × tangerina, Plum, Chayote
  - g10: Cucumber*, Peach, Coconut, Pecan, Chestnut
  - g11: Cucumber*, Coriander, Parsley, Mint, Rosemary, Basil
  - g12: Cucumber*, Onion, Garlic, Spinach, Banana, Broccoli, Cauliflower
- **Bridge crops (appear in >1 group):** Tomato (g1,g3), Lime (g2,g5), Cabbage (g2,g4), Peach (g5,g10). These are *extra* anchors you can chain on, but `stitch_trends_panel.py` must **dedupe** them (keep from the reference group, drop elsewhere) or they'll double-count on merge. `DUPLICATE_CROPS` currently lists only `{cucumber, tomato}` — add `lime, cabbage, peach`.
- **Stitched panel — currently MISSING.** `agri_trends_panel_2016_2025_weekly_FL.csv` is **no longer in the folder** and must be regenerated. And the stitcher only covers part of the data: `stitch_trends_panel.py` has `GROUPS = ["g1","g2","g3"]` and `GROUP_ANCHORS` only for `g2`,`g3`, so even when rerun it yields the old **12-crop** panel (union of g1–g3). **g4–g12 (the other 36 terms) are downloaded but never stitched.** To use everything: set `GROUPS = ["g1",…,"g12"]`, add `g4…g12: ["cucumber"]` to `GROUP_ANCHORS`, extend `DUPLICATE_CROPS` as above, then re-stitch. Do this **before** locking the keyword list.
- **Windowing / pulls:** each group has **19 overlapping ~1-year windows** (`p01…p19`, Jan–Dec and Jul–Jun, overlapping ~6 months) stitched into one weekly series — *not* the 3-window (W1/W2/W3) scheme in `Trends_Pull_Plan_FL_weekly.md`. Crucially, there is **one pull per window, no `pull1/2/3` replicates**, so the **multi-pull noise floor (§5.2) has not been collected** in this layout. See the noise-floor note below.
- **Span (old 12-crop panel):** **523 weekly points, 2015-12-27 → 2025-12-28 — almost exactly 10 annual cycles**, near-zero missingness. Re-confirm span/missingness after regenerating the full 48-crop panel.
- **Constraint that shapes everything below:** you cannot call a Trends API in a loop. Anything *recomputed per fold* must work on the already-downloaded panel; anything needing a *fresh pull* (related-queries, new keywords) is a one-time manual action you archive.

> **Scope warning (new groups).** g9–g12 push beyond Florida specialty crops into **herbs** (coriander, parsley, mint, rosemary, basil), **tree nuts** (pecan, chestnut, coconut), **alliums** (onion, garlic), and general produce (banana, broccoli, cauliflower, spinach). That is fine, but it widens the domain claim. Decide and **pre-register** the in-scope set: either (a) keep the paper "Florida specialty/horticultural crops" and drop out-of-scope terms, or (b) reframe as "consumer food-search interest" and keep them. Also watch term *type* — e.g. `Maize`, `Citrus fruit`, `Citrus × tangerina` look like Trends *Topic*-style entities, not plain search terms; confirm each is a consistent "Search term" so the panel stays comparable.

> **Scale note for ~48 nodes:** a 48-node graph makes the *partial-correlation / graphical-lasso* backbone (§7) essential — raw correlation at this size is overwhelmingly a shared-seasonality graph. Expect to enforce sparsity hard. Re-run the zero-fraction gate after stitching; niche terms (Guava, Okra, Sugarcane, Carambola, Chayote, Chestnut, individual herbs) may fail at FL-weekly and should be flagged or dropped, and that gate result should itself be reported.

One important correction to the robustness doc's framing: it worries about "only ~5 seasonal cycles." You actually have **~10**. That materially weakens the "deep models can't see the yearly cycle / will overfit" concern and makes a 104-week input window and annual-lag features realistic. Say this explicitly in the paper — it is a genuine strength.

---

## 1. Decision summary (read this first)

| Question | Recommendation |
|---|---|
| Which node upgrade to lead with? | **Upgrade B (multi-view node features)** as the backbone, with **Upgrade A (intent decomposition) as the LLM-powered novelty layer on top.** They compose; you don't have to choose. |
| What is the LLM/NLP doing? | **Primarily static, offline feature & structure engineering** — intent classification of related-queries, semantic edge layer, and LLM-assisted (human-verified) agronomic attributes. This is leakage-free, fully reproducible, and defensible. Text-derived exogenous signal is a secondary extension; an LLM *forecaster* is at most an honest baseline. |
| Why is this publishable? | It answers the reviewer's two killer questions (capacity vs. topology; "so what") **and** adds a behavioral/economic layer — does *production-* or *price-intent* search lead *consumption-intent* search? — that pure accuracy papers lack. |

---

## 2. Implementing Upgrade B: the multi-view node feature tensor

Keep **node = crop** (12 nodes). Give each node a multi-channel feature tensor `X[node, time, channel]`. Every channel below is either already in your panel or obtainable **without** new Trends pulls.

**Channel groups (per crop, per week):**

1. **Target + causal lags.** The crop's own Trends value and lags `{1, 2, 4, 8, 13, 26, 52}` weeks. The **52-week (annual) lag is the single most important feature** for these series and your 10-year span supports it.
2. **Calendar / seasonality.** Fourier terms for the annual cycle (`sin/cos` at harmonics `k = 1..4` of period 52) plus sub-annual harmonics. Add **in-season / out-of-season binary flags per crop** from a Florida crop calendar (UF/IFAS) — this is where the LLM helps draft the calendar (§4.3).
3. **Exogenous environment.** Florida weekly weather from **NOAA GHCN-D / Climate Data Online** or **PRISM** (statewide or growing-region average of temp, precip, GDD). Free APIs, one-time bulk download, no per-fold pulling. This is your primary exogenous channel.
4. **Static node attributes (broadcast across time).** Crop family, season class, perishability, typical price tier, substitute/complement group. Used by relational/attention layers and by the community-vs-season interpretability test. LLM-assisted extraction (§4.3).
5. **(Upgrade A layer) Intent sub-channels.** For each crop, the search-intent decomposition (consumption / acquisition / price / production) as separate sub-series — see §3.

> **On Wikipedia pageviews (dropped from the core design).** An earlier draft included English-Wikipedia pageviews as an independent "corroborating view" to separate true interest from Trends sampling jitter. It is cut from the core feature tensor because (i) its job — quantifying Trends' measurement noise — is done directly by the multi-pull **noise floor** (§5.2); (ii) **encyclopedic** reading intent (botanical/reference) does not cleanly track the consumer/market demand this project targets; and (iii) Wikipedia human pageviews are in measurable decline (~8% YoY in 2025, driven by AI search summaries / zero-click), adding a non-crop-specific downward confound. It remains an *optional, free* feature you may slot back into the feature-group ablation; a clean negative result ("encyclopedic attention does not track FL crop demand") is itself reportable. Your authoritative external signal is the **USDA shipment/price linkage (§10)**, not Wikipedia.

> **Practical note on dimensionality.** With 12 nodes × 523 weeks you are data-poor. Do **not** dump 40 raw channels into the model. Group features, regularize hard, and run a **feature-group ablation** (§9) rather than per-feature. Report which *groups* matter; that is the scientific result, not a kitchen-sink input.

---

## 3. Upgrade A as the LLM novelty: search-intent decomposition (realistic version)

This is the highest-novelty, lowest-leakage way to use an LLM, and it fits your manual workflow perfectly.

**The manual acquisition step (one-time).** For each crop, in the Trends UI, download the **"Related queries"** panel (Top + Rising) — the same UI you already use, just a different export. You get a list of query strings like `strawberry picking near me`, `strawberry price`, `strawberry recipe`, `strawberry plants`. Archive these CSVs in `data/related_queries/`.

**The LLM step (offline, deterministic, archived).** Run each query string through an LLM **once** with a fixed prompt that classifies it into one of four intent channels:

- **Consumption** ("recipe", "shortcake", "smoothie")
- **Acquisition / local** ("picking near me", "u-pick", "farm stand")
- **Price** ("price", "cost", "per pound")
- **Production / supply** ("growing", "plants", "season", "harvest")

You then form **intent-channel sub-series** for each crop. Two feasible ways given manual pulls:

- **(a) Curated joint query (recommended, common-scale-safe).** For each crop×intent, pick the representative queries the LLM surfaced and do a *one-time* manual Trends pull, mirroring your existing G1–G8 protocol. See §3.1 for the exact acquisition + scaling recipe — this is where the 5-terms-at-once scaling problem is handled.
- **(b) Weighted reconstruction (no extra pulls).** Approximate each intent sub-series as an intent-share-weighted version of the crop's existing series, where shares come from the LLM-classified related-query volumes. Cheaper, but weaker; use only if (a) is infeasible.

**Why reviewers reward it.** It converts a flat 12-crop graph into a **crop × intent** structure and lets you ask a *demand-economics* question: does production- or price-intent attention lead consumption-intent attention? That is testable with your lead–lag and transfer-entropy edges and is the kind of substance that survives "so what."

**Reproducibility discipline (write this in the appendix):** archive (i) the exact related-query CSVs, (ii) the LLM model+version, (iii) the verbatim prompt, (iv) the full input→label mapping, and (v) a human-verified correction log. The LLM is a *labeling instrument*, and you document it like one. Have a second annotator check a 100-query sample and report inter-rater agreement (Cohen's κ) — this pre-empts "the LLM hallucinated your categories."

### 3.1 Acquiring crop×intent queries and keeping them on scale

This subsection answers two practical questions: **how to get the queries**, and **how to stop the 5-terms-in-one-Compare normalization from destroying the niche intent series.**

**Step 1 — Get the menu, not a time series.** The Trends "Related queries" panel returns a *ranked snapshot list*; its 0–100 numbers are relative within that list and have **no time axis**. Use it only as a candidate-query menu. For each crop: single-term Explore (geo = FL), export Related queries (Top + Rising) → `data/related_queries/`. Feed strings to the LLM to bucket into the four intents.

**Step 2 — Re-pull chosen queries as series via the Compare box.** Only terms typed into Compare produce the weekly series you model. The related-queries export never becomes a feature directly.

**Step 3 — Use the `+` OR operator to beat zeros.** In the Compare box, `strawberry recipe + strawberry shortcake + strawberry smoothie` is treated as **one term** whose volume is the union of its parts. So **each intent channel = one OR-aggregated slot**, not one fragile query. This is the single biggest lever against the zero/quantization problem, because niche intents (`strawberry price`) are exactly the ones that round to 0 at FL-weekly.

**Step 4 — Understand the normalization, then exploit it.** Trends scales *all terms in one Compare* to a single 0–100 where 100 = the highest (term, week) cell in the group. That has a good half and a bad half:

- **Good — within-crop intent is automatically comparable.** One Compare = the 4 intent slots for a single crop → those four already share one scale. That is all you need for the headline lead–lag / transfer-entropy question (*does price- or production-intent lead consumption-intent for strawberry?*). **No anchor required — do not over-engineer this part.**
- **Bad — cross-crop intent needs a bridge, and a dominant anchor backfires.** To put `strawberry-consumption` and `blueberry-consumption` on one scale you need an anchor, but a high-volume anchor (Cucumber/Tomato) swamps niche intents and quantizes them to noise. **Fix: use the crop's own main term as the 5th slot and as the bridge:**

```
Compare (one crop) = [ crop_main , intent_consumption , intent_acquisition , intent_price , intent_production ]
```

`crop_main` is already on your Cucumber-anchored panel scale, so rescale that crop's four intents by the `crop_main` ratio to land them on the panel scale — one more hop of the same chaining logic as G1–G8. Fits exactly in the 5-term limit, and the intents sit *below* their own crop term (their natural volume range), avoiding the dominant-anchor quantization.

**Step 5 — Gate for sparsity; scope honestly.** Even with OR-grouping, many intent×crop cells are too sparse at FL-weekly (`guava price`, `papaya picking` ≈ all zeros). Apply the same zero-fraction gate as crops (drop > ~20–30% zeros) and report survivors. Expect intent decomposition to be a **focused sub-study on the high-volume crops**, not all 30 — reviewers prefer a clean intent analysis on ~8 crops to a zero-riddled one on 30. Two documented escape hatches if FL-weekly is too thin: aggregate the intent layer to **monthly**, or pull intent at **US-national** geo (denser) while keeping FL for the main series, flagging the geo mismatch. Within-crop lead–lag survives either choice.

**Step 6 — Multi-pull intent groups too** (3 pulls × different days) so the intent series get the same noise-floor treatment as the crop panel.

### 3.2 Worked example — OR-query groups per intent

Representative high-volume crops likely to survive the sparsity gate. Each cell is one OR-aggregated Compare slot; the row's `crop_main` (the bare crop term, on the panel scale) is the bridge. **Verify each query's intent against live Related-queries before locking** — these are starting templates, not ground truth, and FL-specific terms (`u-pick`, `picking near me`) matter.

| Crop (`crop_main`) | Consumption | Acquisition / local | Price | Production / supply |
|---|---|---|---|---|
| **Strawberry** | strawberry recipe + strawberry shortcake + strawberry smoothie | strawberry picking + strawberry picking near me + u pick strawberries | strawberry price + strawberries per pound | growing strawberries + strawberry plants + strawberry season |
| **Watermelon** | watermelon recipe + watermelon salad + watermelon juice | watermelon near me + watermelon stand | watermelon price + price of watermelon | growing watermelon + watermelon plant + watermelon season |
| **Tomato** | tomato recipe + tomato sauce + tomato soup | tomatoes near me + farm tomatoes | tomato price + tomatoes per pound | growing tomatoes + tomato plants + when to plant tomatoes |
| **Blueberry** | blueberry recipe + blueberry muffins + blueberry pie | blueberry picking + blueberry picking near me + u pick blueberries | blueberry price + blueberries per pound | growing blueberries + blueberry bushes + blueberry season |
| **Avocado** | avocado recipe + avocado toast + guacamole | avocados near me | avocado price + price of avocados | growing avocado + avocado tree + avocado season |
| **Peach** | peach recipe + peach cobbler + peach pie | peach picking + peaches near me + u pick peaches | peach price + peaches per pound | growing peaches + peach tree + peach season |
| **Lemon** | lemon recipe + lemonade + lemon cake | lemons near me | lemon price + lemons per pound | growing lemons + lemon tree + meyer lemon tree |
| **Potato** | potato recipe + mashed potatoes + baked potato | potatoes near me | potato price + potatoes per pound | growing potatoes + planting potatoes + seed potatoes |
| **Pumpkin** | pumpkin recipe + pumpkin pie + pumpkin bread | pumpkin patch + pumpkin patch near me | pumpkin price + price of pumpkins | growing pumpkins + pumpkin plant + pumpkin seeds |
| **Sweet potato** | sweet potato recipe + sweet potato pie + sweet potato fries | sweet potatoes near me | sweet potato price + sweet potatoes per pound | growing sweet potatoes + sweet potato slips + planting sweet potatoes |

For lower-volume crops (Guava, Okra, Sugarcane, Eggplant, Radish, Zucchini, Celery, Carrot, etc.), build the same four OR-slots from their Related-queries menus, run the gate, and keep only those that pass — likely a 0–2-intent subset each. Document every dropped cell.

> **Acquisition budget.** ~8–12 viable intent crops × 1 Compare each × your window scheme (19 overlapping windows, as in g1–g12) × 3 pulls for the noise floor. Front-load it, archive everything, then never touch Trends again during modeling. (If 19 windows × 3 pulls per crop is too heavy, run intent decomposition on the W1/W2/W3 three-window scheme instead — coarser stitch, far fewer downloads.)

---

## 4. Recommended LLM/NLP integration (since you were unsure)

Use the LLM in three **static, offline** roles. None touches the test fold; all are archivable and reproducible. This is the defensible core. Two optional extensions follow.

### 4.1 Intent classification (the novelty) — §3 above. **Do this.**

### 4.2 Semantic / co-search edge layer
Build one layer of the multiplex graph from **text**, not co-movement:
- Embed each crop's identity + its related-query set with a sentence-embedding model (e.g. `sentence-transformers` / a small instruct-embedding model). Cosine similarity → a **semantic adjacency** `A_sem`.
- This captures behavioral relatedness ("lime"↔"key lime pie"↔"cocktail") that price/co-movement edges miss.
- Because it is derived from text + a frozen model, it is a **static prior** — compute once, no per-fold refit, no leakage. Document the model version.

### 4.3 LLM-assisted agronomic attributes & expert graph
Replace tedious manual UF/IFAS lookups with **LLM-drafted, human-verified** structured tables:
- Per crop: family, FL season window (start/end weeks), perishability class, substitute/complement set, commodity group.
- These feed (i) the static node attributes (§2.5), (ii) the **agronomic edge layer** `A_agro` (shared season window / commodity group / substitute links), and (iii) the in/out-of-season flags (§2.2).
- **Critical:** the LLM *drafts*; you *verify against UF/IFAS and USDA before use*. Report it as "LLM-assisted curation, expert-verified," never as ground truth. Keep the diff between LLM draft and verified final as an appendix artifact.

> Why static-and-offline is the right call: an LLM in the *forecast loop* creates (a) leakage risk, (b) non-determinism that breaks reproducibility, and (c) a reviewer magnet ("is this just data contamination from pretraining?"). Used as an offline labeling/encoding instrument, the LLM is clean, cheap, and adds genuine novelty.

### 4.4 Optional extension — text-derived exogenous signal
If you want more: build event/sentiment **time-series** features from free text sources and add them as extra node channels — e.g. weekly counts of crop-relevant **USDA narrative reports**, **Wikipedia pageview spikes**, or **news headline** mentions (freeze/hurricane/recall events that move FL specialty-crop attention). More novel, but more acquisition and more leakage surface (these must be timestamped and lagged causally). Treat as a v2 feature, not a v1 dependency.

### 4.5 Optional baseline — LLM-as-forecaster
You *may* add a zero-shot LLM/foundation time-series forecaster (TimeLLM-/Chronos-/Moirai-style) to the baseline ladder. Frame it honestly: reviewers are skeptical and pretraining-contamination concerns are real for a public signal like Trends. Include it for completeness and to show your GNN's gains aren't trivially matched — not as a headline.

---

## 5. Data preprocessing

All steps fit **on the training portion of each rolling-origin fold only.**

1. **Assemble & calibrate** the panel via the anchor-rescale + stitch scripts (Cucumber primary anchor; window overlaps). **First make the stitcher cover everything:** set `GROUPS = ["g1"…"g12"]`, add `g4…g12: ["cucumber"]` to `GROUP_ANCHORS`, and add `lime, cabbage, peach` to `DUPLICATE_CROPS` (alongside `cucumber, tomato`) — otherwise only g1–g3 (12 crops) stitch and bridge crops double-count. Regenerate `agri_trends_panel_…csv` (currently absent). The raw pulls in `data/g*` are the archived dataset (the target is not perfectly reproducible).
2. **Noise floor (do before modeling) — gap to close first, and now the *sole* mechanism for the noise question.** With Wikipedia dropped (§2), this is the only thing separating true interest from Trends sampling jitter, so treat it as required rather than optional. It needs *repeat pulls of the same window on different days*, but the current `data/g*` layout has **one pull per window (no replicates)**, so the noise floor cannot yet be estimated. Two options: (a) **collect** a small multi-pull sample — re-download a handful of representative group×window cells 3× on different days — and compute per-crop across-pull variance; or (b) as a weaker proxy, use the **disagreement on the ~6-month overlaps** between adjacent windows (`p_k` vs `p_{k+1}`) as a lower-bound jitter estimate. Either way, report skill *relative to the floor*; no model may honestly claim error below it. Document which option you used.
3. **Calendar regularization.** Ensure a clean weekly index, handle the weekly/monthly boundary at the stitch seam, forward-fill at most isolated gaps (flag any).
4. **Zero / low-interest handling.** Your panel is clean (near-zero low-interest fraction), so no crop needs dropping — state the zero-fraction gate you applied and that all 12 passed.
5. **Stationarization for graph construction only.** Compute **STL remainders** (remove trend + annual season) — these residuals, not raw series, feed the statistical edges (§7). Keep raw+features for the forecaster.

---

## 6. Normalization

- **Forecasting input:** per-crop scaling **fit on train only**, refit each fold. Prefer a robust scaler (median/IQR) or `log1p` over min-max, given the bounded 0–100 range and occasional spikes. Because the panel is already anchor-calibrated to a common scale, per-crop scaling is for optimization stability, not comparability.
- **Targets & metrics:** evaluate with **scale-free MASE/RMSSE** (§8), so you are not hostage to a particular normalization.
- **Graph inputs:** z-normalize STL remainders before correlation/DTW so edge weights aren't dominated by high-variance crops.
- **Leakage rule:** every scaler, decomposition, PCA, and intent-share weight is a *fitted object* tied to a fold. Never fit on the full sample.

---

## 7. Graph construction (multiplex, validated, train-only)

Build a **multi-relational (multiplex) graph**, not a single adjacency. Data-driven layers are refit per fold on STL remainders; prior layers (semantic, agronomic) are static.

| Layer | Method | Refit per fold? | Role |
|---|---|---|---|
| `A_part` (**primary**) | Graphical-lasso precision matrix on STL remainders → direct conditional dependence | Yes | Removes spurious shared-season edges |
| `A_lag` | Directed argmax cross-correlation (max lag 13) | Yes | Lead–lag (strawberry→blueberry question) |
| `A_te` | Transfer entropy (PyIF/IDTxl) | Yes | Nonlinear directed information |
| `A_agro` | LLM-assisted, expert-verified expert graph | No (prior) | Knowledge anchor / interpretability |
| `A_sem` | Sentence-embedding cosine over crop + related-query text | No (prior) | Behavioral relatedness from text |
| `A_learned` | End-to-end adaptive adjacency inside MTGNN/AGCRN/GWN | Learned | Discovery; validated vs `A_agro`/`A_sem` |

**Statistical validation (the credibility step).** For each data-driven layer, keep an edge only if it survives a **null model** — stationary-bootstrap (Politis–Romano) confidence intervals or phase-randomized/IAAFT surrogates — with **FDR control (q = 0.05)**. Report the surviving edge count and a defended-edge-by-edge graph. With only ~250–500 weekly points this is what separates a real graph from a decorative one.

---

## 8. Models, baselines, and evaluation

### 8.1 Baseline ladder (the capacity controls reviewers demand)
1. **Seasonal-naïve** (the brutal bar) and **SARIMA / ETS / Theta** — local univariate.
2. **VAR** — classical multivariate.
3. **Global deep, no graph** — N-HiTS, PatchTST, TiDE, DeepAR (via NeuralForecast/Darts), trained across all crops. Isolates cross-series pooling.
4. **Capacity-matched global no-graph backbone** — your ST-GNN's exact backbone with message passing **off**. The single most important control.
5. **Identity-graph GNN** (`A = I`) — sanity floor.
6. **Random/permuted-graph GNN** — degree-preserving edge shuffles, many permutations → a **null distribution of accuracy**.

### 8.2 GNN models (the contribution)
- **Learned-graph:** MTGNN, AGCRN, Graph WaveNet (the learned-vs-fixed story).
- **Fixed/directed:** STGCN, DCRNN on `A_part` / `A_lag`.
- **Relational/multiplex GNN** (RGCN-style over the stacked layers) — carries the multi-edge claim.
- **Probabilistic output** (quantile or DeepAR-style) so you can report pinball loss + interval coverage — well received at ISF and usually skipped by competitors.

### 8.3 Metrics & protocol
- **Primary:** MASE (scaled vs seasonal-naïve). **Secondary:** RMSSE. **Report-only:** MAE/RMSE, sMAPE (note instability near low weeks). **Probabilistic:** pinball loss + coverage. **Directional:** Pearson/Spearman of forecast vs actual.
- **Rolling-origin CV**, multiple origins; horizons `h ∈ {1, 4, 8, 13}`; report mean ± std across folds and horizons.
- **Significance:** Diebold–Mariano per model pair; **Friedman + Nemenyi/MCB** across the 12-crop panel with a **critical-difference diagram** (the ISF-expected figure).
- **Multiple seeds** per deep model; report variability.
- **Pre-register** crop list, intent-keyword list, and metrics before touching test data.

### 8.4 The decisive experiment
Hold model capacity fixed; vary only edges. Headline result: **does the structured multiplex graph beat the permutation-null distribution by a statistically significant DM margin?** If yes → topology, not capacity, carries the signal (clean, publishable). If no → an honest "when does graph structure actually help search-interest forecasting?" result, which top venues also value.

---

## 9. Ablation studies

Run these as the evidentiary core of the paper:

1. **Edge-layer ablation.** Remove one relation at a time (`A_part`, `A_lag`, `A_te`, `A_agro`, `A_sem`) and measure the accuracy drop → attributes value to each relation.
2. **Graph-permutation null.** Degree-preserving shuffles × many seeds; DM test of the real graph vs the null distribution (§8.4).
3. **Feature-group ablation.** Toggle each Upgrade-B group: annual-lag block, Fourier/season flags, weather, intent sub-channels (and, optionally, Wikipedia pageviews if you reinstate them). Confirms the multi-view design earns its complexity.
4. **LLM/NLP ablation (your novelty's evidence).**
   - With vs. without the **semantic edge layer** `A_sem`.
   - With vs. without **intent decomposition** (flat crop graph vs crop×intent).
   - **LLM-assisted agronomic graph vs. hand-coded** — shows the LLM curation didn't degrade quality.
5. **Capacity-matched control** (global no-graph) — the headline capacity isolation.
6. **Input-window ablation.** 12 vs 52 vs 104 weeks — your 10-year span makes the long window viable; show seasonal context matters.
7. **Noise-floor comparison.** Plot every model's error against the per-crop floor; flag any "wins" that sit below the floor as overfitting jitter.

---

## 10. External validity (the "so what" — highest leverage)

Forecasting a Trends index better is not, alone, an IJF contribution. **Forecasting demand attention that demonstrably leads real market outcomes is.** Pull free **USDA-NASS QuickStats / AMS** weekly shipment & price series for the FL specialty crops, then test whether **graph-derived** Trends forecasts lead USDA shipments/prices (Granger/TE) better than **non-graph** signals — and better than the permutation null. If the graph-informed search signal leads real outcomes and the null does not, you've shown the network captures economically meaningful demand structure. That sentence is your abstract.

---

## 11. Build order (revised for your manual workflow)

1. **Lock the target.** Regenerate the full 48-crop panel (extend `GROUPS`/`GROUP_ANCHORS`/`DUPLICATE_CROPS` for g4–g12); decide and pre-register the in-scope crop & intent-keyword list; collect (or proxy) the multi-pull noise floor and report it; confirm the calibrated panel.
2. **One-time LLM acquisitions.** Download related-queries CSVs; run the offline intent classifier; build `A_sem`; draft+verify agronomic attributes. Archive every artifact.
3. **Free programmatic features.** NOAA/PRISM weather → merge into the feature tensor.
4. **Build validated multiplex graphs** per fold; run the community-vs-season interpretability test.
5. **Stand up the full baseline ladder**, including the capacity-matched global no-graph model and the permutation-null harness.
6. **Run the decisive experiment** with DM tests + critical-difference diagram; report skill relative to the noise floor.
7. **External validity:** Trends-graph forecasts → USDA shipments/prices.
8. **Write for IJF/ISF:** lead with the methodological + economic claim; full reproducibility appendix (raw pulls, LLM prompt/version/labels, code, seeds); honest small-data and noisy-target discussion; note the **10-year / ~10-cycle** strength.

---

## 12. Reproducibility & leakage checklist (LLM-aware)

- Graph, decomposition, scalers, PCA, intent-share weights: **train-fold only**, refit per fold.
- Causal feature engineering only — past-window rolling stats; no centered windows.
- Static LLM artifacts (`A_sem`, agronomic attributes, intent labels) computed **once, offline**, version-pinned, archived with prompt + model version — they are priors, not fold-fitted, and must not encode future information.
- Same splits, seeds, input window, horizons across **all** models.
- Archive exact Trends pulls (date, geo, term, window) + all LLM inputs/outputs — the raw pulls and the LLM label set together *are* the dataset.
- Second-annotator check on an intent-label sample; report κ.

---

### Stack
Python · pandas/statsmodels/pmdarima · NeuralForecast & Darts (global deep + probabilistic baselines) · PyTorch Geometric / PyG-Temporal (ST-GNNs) · scikit-learn GraphicalLasso · dtaidistance/tslearn (DTW) · PyIF/IDTxl (transfer entropy) · NetworkX + Leiden (graph + communities) · sentence-transformers (semantic edges) · NOAA CDO / PRISM · USDA QuickStats / AMS · Google Trends Anchor Bank (calibration).
