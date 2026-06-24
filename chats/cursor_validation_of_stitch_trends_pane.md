# Validation of stitch_trends_panel.py
_Exported on 6/22/2026 at 16:52:27 EDT from Cursor (3.7.19)_

---

**User**

how do I validate or provide reasoning to whatever I have done in the @stitch_trends_panel.py is correct while publishing a paper? read through the files in the 'design docs; to get the context and then give me a succint response.

---

**Cursor**

I'll read the script and the design docs to understand what you've built and how to justify it.

I'll read the design doc to get the full research context.

Here's how to defend the stitching in `stitch_trends_panel.py` as a validated, reproducible procedure for an IJF/ISF submission. The core idea: your stitching is an *estimation* step, so treat it like one — quantify its error, show it's robust to choices, and corroborate it against an independent signal.

## 1. The cheap, decisive validation: held-out overlap error
Your method already creates a natural validation set — the ~26-week overlaps you stitch on. Exploit them:

- For each join `p_k → p_{k+1}`, after computing the scale factor and aligning, measure the **reconstruction error on the overlap weeks** (the two windows should now agree). Report MAPE / median relative error and a Bland–Altman plot of `running_panel` vs `scaled_incoming` on shared dates. Small, unbiased residuals = the ratio-of-means assumption (a single multiplicative scale relates the two pulls) holds.
- This directly answers a reviewer's first question: *"how do you know the seams are right?"* You show the seam residual distribution.

## 2. Show robustness to the estimator choice (sensitivity analysis)
You made two specific modeling choices that a reviewer will probe. Show the panel barely moves under alternatives:

- **Phase 1 estimator:** ratio-of-means (`sum/sum`). Re-run with (a) median-of-ratios, (b) Theil–Sen / log-OLS regression through the origin on overlap pairs, (c) per-crop factors instead of pooled. Report correlation / max relative deviation of the resulting panels. If they agree, the choice is immaterial — say so.
- **Phase 2 anchor choice:** you anchor everything to g1 via cucumber. Test (a) a different reference group, (b) using the bridge crops (tomato, lime, cabbage, peach) as alternative/extra anchors, (c) leave-one-anchor-out. Stable rankings/levels = the cross-group calibration isn't an artifact of one anchor.

## 3. Benchmark against the established method (Google Trends Anchor Bank)
Your own stack list (design doc §Stack) cites **Google Trends Anchor Bank (G-TAB)**. That is the peer-reviewed, citable calibration method (West, 2020). Either:
- run G-TAB on a subset and show your simpler ratio-of-means stitch reproduces it within tolerance, **or**
- explicitly frame your method as a lightweight, overlap-based variant and cite G-TAB as prior art.

Reviewers reward "we validated our calibration against the published standard."

## 4. Independent corroboration (the strongest evidence)
A within-data consistency check can't rule out that the *whole* signal is a Trends sampling artifact. The design doc already prescribes the fix in §2.3 / §9.3: **Wikipedia pageviews** (programmatic, not subject to Trends renormalization). After stitching, correlate each crop's stitched series against its weekly Wikipedia pageviews. High correlation = your stitched panel tracks a real, independently-measured interest signal, not stitching noise.

## 5. Quantify the noise floor you're stitching against
The design doc flags this as an open gap (§0, §5.2): your layout has **one pull per window, no replicates**, so you can't yet say how much of the seam residual is irreducible Trends jitter. Do one of:
- collect 3× repeat pulls on a few representative window cells → per-crop across-pull variance, **or**
- use the **overlap disagreement (§1 above) as a lower-bound jitter estimate**.

Then state: stitching error ≤ noise floor (or report the ratio). No downstream forecast may claim skill below this floor.

## Two correctness bugs to fix first (or your validation validates the wrong panel)
The design doc explicitly calls these out, and the code confirms them:

1. **Phase 3 "earlier group wins" is order-dependent and `g1` isn't guaranteed first.** In `main()` you iterate `[g for g in groups if g != ref_group]` and keep whichever group is seen first for a duplicate. For bridge crops that are *not* in `g1` (lime: g2/g5; cabbage: g2/g4; peach: g5/g10), the "keep from reference group" intent silently becomes "keep from lowest-numbered group." That's defensible, but it is **not** what the docstring claims, and you must document the actual dedup rule you report.

2. **All groups are rescaled using cucumber only.** `group_anchors` maps every group to `["cucumber"]`. The design doc notes cucumber can *swamp* niche crops and that bridge crops are available as extra anchors. For groups containing a bridge crop, adding it as a second anchor (your `anchor_scale_factor` already averages multiple) makes Phase 2 more robust — and gives you the leave-one-anchor-out test in §2 for free.

---

### What to write in the paper (one paragraph)
> "Half-year windows were stitched per crop via overlap ratio-of-means rescaling, pooled across crops for robustness (Phase 1), then groups were placed on a common scale through the Cucumber anchor present in all 12 groups, validated by bridge crops (Phase 2). We validate the calibration three ways: (i) held-out reconstruction error on the ~26-week window overlaps (median relative error X%, no systematic bias, Fig. N); (ii) insensitivity to the scale estimator and anchor choice (max panel deviation Y% across alternatives, App. N); and (iii) external corroboration against Wikipedia pageviews (mean ρ = Z). Stitching residuals are at/below the Trends multi-pull noise floor, against which all forecasting skill is reported."

Want me to add the overlap-reconstruction-error reporting and the bridge-crop anchors directly into `stitch_trends_panel.py`?

---

**User**

I want you to layout practical validation steps. should I do this step now after I complete the entire implementation pipeline. give me an honest review 

---

**Cursor**

Let me look at what you already have in the validation file before I give you the honest answer.

Read both — the `.py.py` is gone, and your current `validate_panel.py` does **structural sanity only** (shape, gaps, zero-fraction, value range, dupes). That's necessary but it does **not** validate that the stitching is *correct*. It would happily pass a perfectly-shaped but miscalibrated panel.

## Honest review of your timing question

**No — do not defer stitching validation to the end of the pipeline.** This is the one place where "validate at the end" is genuinely dangerous, and here's the honest reasoning:

- The panel is the **foundation every downstream number sits on**. If a seam is miscalibrated by 1.5×, your STL remainders, graphical-lasso edges, lead–lag directions, and forecast errors are all contaminated — and you'd discover it *after* building the GNN, baselines, and ablations. That's weeks of work resting on an unverified assumption.
- Stitching validation is **cheap and self-contained** (it runs on the overlaps you already have — no new pulls, no model). There's no efficiency argument for waiting.
- The design doc itself says "**Lock the target**" is build-order step 1, and "do this *before* locking the keyword list." You're already aligned with that; the validation is what *lets* you lock it.

So the correct mental model is **two validation gates, not one**:

| Gate | When | Validates | Blocks if it fails |
|---|---|---|---|
| **A — Data/calibration** | NOW, before any modeling | stitching is correct, panel is real | the entire pipeline |
| **B — Model/scientific** | END, after pipeline | the graph beats the null, signal leads markets | the *claims*, not the data |

Doing B now is impossible (no models yet). Doing A at the end is the actual mistake. They are not the same step and shouldn't be bundled.

**The one caveat where you're partly right:** don't *gold-plate* gate A now. Run the decisive checks, lock the panel, move on. The exhaustive sensitivity sweep can be tightened up later for the appendix once you know the panel matters. Below I mark what's "must do now" vs "appendix-later."

---

## Gate A — practical steps to run NOW (before modeling)

You can add these to `validate_panel.py`, but most need the **raw windows**, not just the final CSV — so a companion `validate_stitch.py` next to it is cleaner.

**A0. Structural sanity — DONE.** Your current file. Keep it. Just also assert the value range is *exactly* `[0,100]`-ish and that the dedup left no duplicate crop columns (it already does).

**A1. Held-out overlap reconstruction error — MUST DO NOW (the decisive one).**
This is the single check that proves the seams are right. For each join `p_k → p_{k+1}`, after applying the scale factor, compare the two windows on their shared weeks:
- report median & 90th-pct **relative error** per seam,
- report **bias** (mean signed error — should be ≈0; nonzero means a single multiplicative scale is the wrong model for that seam),
- flag any seam above a threshold (e.g. >15% median).
This requires modifying `stitch_windows` to *return the per-seam residuals*, or re-deriving them in the validator. Honestly, the cleanest path is to have the stitcher emit a `seam_report.csv`.

**A2. Anchor leave-one-out (Phase 2) — MUST DO NOW, cheap.**
You currently anchor every group to g1 via cucumber only. Recompute each group's cross-scale factor using bridge crops (tomato/lime/cabbage/peach) where available, and report the spread. If cucumber-only vs. multi-anchor disagree by a lot, your cross-group levels are fragile and you need to fix Phase 2 *before* modeling.

**A3. Estimator sensitivity — LIGHT version now, full in appendix.**
Now: re-run Phase 1 once with median-of-ratios instead of ratio-of-means and report max panel deviation. If it's small (<~5%), you're done — note it and move on. The full Theil–Sen / per-crop / log-OLS sweep is appendix-later.

**A4. Independent corroboration (Wikipedia pageviews) — DO NOW-ISH (high value).**
This is the strongest evidence the panel is *real signal, not Trends artifact*, and it's programmatic/free. I'd pull it during gate A rather than at the end, because if a crop's stitched series *doesn't* correlate with its Wikipedia interest, that's a data problem you want surfaced before modeling. You don't need the full feature merge yet — just the correlation check.

**A5. Noise-floor decision — DECIDE NOW, collect now if you're going to.**
The design doc flags this as your biggest open gap: you have one pull per window, no replicates. You must decide now between (a) collecting a few 3× repeat pulls, or (b) using the A1 overlap disagreement as a lower-bound proxy. Decide now because option (a) means going back to the Trends UI — and the doc's whole philosophy is "front-load all manual pulls, then never touch Trends again." Discovering at the end that you need replicate pulls is the exact trap to avoid.

---

## Gate B — defer to END of pipeline (don't touch now)
- Forecast skill vs. seasonal-naïve, MASE/RMSSE, DM tests, critical-difference diagram.
- Graph-permutation null (does the real graph beat shuffled graphs?).
- Edge/feature/LLM ablations.
- External validity: do Trends-graph forecasts lead USDA shipments/prices?

These validate the *scientific claims* and require the trained models. Running them now is impossible and trying to scaffold them now is premature.

---

### Bottom line
Do **gate A now** — specifically A1, A2, A5-decision are non-negotiable before you lock the panel; A3-light and A4 are high-value and cheap. Defer **gate B** to the end. The instinct to "validate everything at the end" is wrong *only* for the data-calibration layer, because that layer is load-bearing for everything else and free to check now.

Want me to (1) extend the stitcher to emit a per-seam `seam_report.csv`, and (2) write `validate_stitch.py` implementing A1–A4? That's the concrete "do it now" deliverable.

