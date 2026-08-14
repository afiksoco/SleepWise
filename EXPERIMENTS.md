# SleepWise — Model Experiment Log

Running ledger of models we train so results are reproducible and comparable.
Every row is grouped-by-subject 5-fold CV (no subject leakage). Deep metrics are
at each model's own operating threshold (calibrated to the training deep prior).

> **How to read "lower" numbers:** once we pool Walch + DREAMT, the test set is
> harder and more varied (two cohorts, two devices) than Walch-only. A drop vs the
> Walch-only baseline is a *more honest* estimate, not necessarily a regression.
> That's why every combined model reports **per-cohort** metrics too.

---

## Datasets

| Name | Subjects | Signals | Deep % | Notes |
|---|---|---|---|---|
| **Walch 2019** (`data/sleep-accel-dl`) | 31 | HR, accel | ~12% | healthy; PSG labels incl. REM; **no temp/HRV** |
| **DREAMT** (`data/dreamt`, Empatica E4) | ~69 files | HR, accel, **TEMP, IBI(→HRV)**, EDA, BVP | ~3.5% | apnea cohort; PSG labels incl. REM; low Deep |

DREAMT columns: `TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI, Sleep_Stage, {apnea flags}`
Sleep_Stage values: `W, N1, N2, N3, R, P(prep/unknown)` → drop P/missing.

---

## Baseline (current deployed lineage)

### B0 — `walch_binary_v2` (Stage-1, deployed lineage)
- **Data:** Walch only. **Features:** 33 causal (HR/accel; temp removed). **Model:** Dense NN → TFLite.
- **Scheme:** binary_n3 (Deep=N3 vs Light=rest). **Threshold:** 0.332 (calibrated to 12% prior).
- **Grouped 5-fold OOF (n=13,214, deep prev=12.0%):**

| Metric | Value |
|---|---|
| Deep precision | **0.354** |
| Deep recall | **0.830** |
| Deep F1 | **0.496** |
| TP / FP / FN / TN | 1314 / 2400 / 269 / 9231 |

*(Interpretation: catches 83% of true deep sleep; ~1 in 3 "deep" calls is correct.
For a smart alarm this recall-favoring trade-off is intentional — we'd rather not
wake someone in deep sleep.)*

---

## Stage 3 — combined Walch + DREAMT, all four signals

Goal: **one** model that uses HR + accel always, and temp + HRV when present
(DREAMT training + the live Samsung watch), via **missingness indicators**
(`has_hrv`, `has_temp`). Precedent: U-Sleep (multi-cohort), missing-modality learning.

Plan / models to fill in:
- [ ] **C1 — XGBoost** (native NaN handling): honest accuracy ceiling; feature importances.
- [ ] **C2 — Masked Dense NN → TFLite**: deployable; missing feats zeroed + presence-mask inputs.
- [ ] Both: grouped 5-fold by subject **across both cohorts**; class weights; per-cohort metrics.
- [ ] Binary Deep/Light **and** 3-class (Wake/NREM/REM) — REM is the expected real gain.

### Prediction (recorded before training, checked below)
- Binary Deep: **modest/no gain** (HR+accel already carry most deep signal; DREAMT's low-Deep may drag it).
- REM: **real gain** — model produces a usable REM track for the first time; moderate (not PSG-grade) quality.
- Pooled numbers may read lower than Walch-only B0 — that's a harder, more honest test.

### C1 — XGBoost, combined Walch+DREAMT, 40 feats (HR/accel + temp + HRV), grouped 5-fold (2026-08-14)
Combined: 49,825 epochs / 100 subjects (13,214 Walch + 36,611 DREAMT). HRV on 56% of rows (77% of DREAMT).

**A. Binary Deep/Light** (threshold calibrated to Walch recall≈0.83, apples-to-apples with B0):

| Cohort | Deep P | Deep R | Deep F1 | n_deep |
|---|---|---|---|---|
| **walch** | **35.4%** | **83.0%** | **0.497** | 1583 |
| dreamt | 11.4% | 16.8% | 0.136 | 792 |
| pooled | 29.7% | 60.9% | 0.399 | 2375 |
| **B0 (baseline)** | 35.4% | 83.0% | 0.496 | — |

→ **Walch-cohort Deep is IDENTICAL to B0.** Pooling + temp + HRV neither helped nor hurt the alarm's
Deep decision. No regression — a bigger, more diverse training set gives the same Walch performance.
(DREAMT Deep is poor: only 2.2% Deep prevalence, apnea-fragmented — expected.)

**B. 4-class REM ablation — DOES temp+HRV enable REM?** (per-cohort P/R; REM only meaningful on DREAMT, which has both REM labels AND HRV)

| Features | DREAMT REM P | DREAMT REM R | DREAMT REM F1 |
|---|---|---|---|
| HR+accel only (33) | 27.8% | 49.4% | 0.356 |
| **+temp+HRV (40)** | **30.3%** | **43.2%** | **0.356** |

→ **Time-domain HRV did NOT improve REM** (identical F1; +precision, −recall, net wash). My "REM is the
real gain" prediction was **too optimistic** for the features I used.

**Feature importance (binary Deep, gain):** top features are accel/HR rolling stats. temp share **7.2%**,
HRV share **9.4%** — used, but modestly; `hrv_pnn50`, `temp_mean`, `hrv_ibi_mean` crack the top ~10.

**Verdict:** the combined model is a **safe, defensible drop-in** (Walch performance preserved, multi-cohort
story, U-Sleep precedent) but temp+HRV are **not yet earning their keep** — especially for REM.

**Likely reason & next experiment (C3):** I used only *time-domain* HRV (RMSSD/SDNN/pNN50/IBI). The classic
REM discriminator is *frequency-domain* **LF/HF ratio**, plus **per-subject HRV normalization** (absolute RMSSD
varies hugely between people; a personalized z-score, like our `hr_z`, is what the model needs). C3 = add LF/HF
+ personalized HRV/temp baselining, re-run the REM ablation.

### C3 — XGBoost, combined, 47 feats (+ LF/HF frequency-domain HRV + personalized per-subject z-scores) (2026-08-14)

**A. Binary Deep/Light** — walch cohort **35.1% / 83.0% / F1 0.493** (still ≈ B0; DREAMT Deep improved 11.4→15.5%).
**B. DREAMT REM ablation:**

| Features | REM P | REM R | REM F1 |
|---|---|---|---|
| HR+accel only (33) | 27.8% | 49.4% | 0.356 |
| **C3 (47, +LF/HF+pers)** | **29.7%** | **41.7%** | **0.347** |

→ **Still no REM gain.** LF/HF + personalization raised HRV's model usage (gain share 9.4%→**16.5%**, `hrv_hf`
now a top-10 feature) and nudged **Deep** up slightly on DREAMT — but REM F1 is **unchanged (~0.35)**.

**Conclusion (well-tested, defensible negative result):** On Walch+DREAMT, wrist HRV/temp — even with
frequency-domain LF/HF and personalized baselining — **does not improve REM** over HR+accel, and gives
only a marginal Deep bump. Likely causes: DREAMT is an *apnea* cohort (autonomic/HRV signal disrupted),
E4 wrist-PPG IBI is noisy, and 60–120 s windows are short for stable LF/HF. REM's usable signal (HR
variability) is already captured from HR alone.

| ID | Data | Feats | Model | Deep P/R (walch) | DREAMT REM P/R (F1) | Status |
|---|---|---|---|---|---|---|
| B0 | Walch | 33 | NN/TFLite | 35.4 / 83.0 | — | deployed lineage |
| C1 | Walch+DREAMT | +temp+HRV (time) | XGBoost | 35.4 / 83.0 | 30.3 / 43.2 (.356) | done — no REM gain |
| C3 | Walch+DREAMT | +LF/HF + personalized | XGBoost | 35.1 / 83.0 | 29.7 / 41.7 (.347) | done — no REM gain |

**Net for the project:** the *combined multi-cohort* model is a safe drop-in (preserves the alarm exactly,
100-subject/2-device training set, U-Sleep precedent). The 4-stage report is feasible from HR+accel
(REM ~28–46% precision). HRV's honest verdict on this data = **doesn't earn its keep for staging** —
a legitimate finding to report, not a failure to hide.

**Remaining lever for a cleaner REM *track* (not per-epoch accuracy):** temporal modeling — REM occurs in
~90-min-cycle blocks, so Viterbi/HMM smoothing (cheap, deployable) or a GRU over epochs would make the
report's stage track more coherent. Does not change the HRV verdict.

---

## DEPLOY test — can temp+HRV go INTO the on-device alarm model? (2026-08-14)
`build_deploy_model.py`: masked Dense NN → TFLite, 45 feats (causal HR/accel + real temp + time-domain HRV
+ personalized z + has_temp/has_hrv), **modality dropout 0.35**. The watch is ALWAYS masks=1, so the honest
operating point is measured with masks forced to 1.

| Condition | Walch Deep precision | Walch Deep recall |
|---|---|---|
| masks=0 (Walch's natural state) | 28.1% | 83.0% |
| masks=1, same threshold | 36.0% | 57.5% (recall collapse) |
| **masks=1, threshold recalibrated = DEPLOYMENT point** | **23.7%** | **83.0%** |
| **B0 / HR+accel-only (no temp/HRV)** | **35.4%** | **83.0%** |

→ **Putting temp+HRV into the alarm model makes it WORSE on the watch: 23.7% vs 35.4% Deep precision** at the
same recall. Cause: "has temp/HRV = 1" only ever co-occurred with the low-Deep DREAMT cohort (+ E4-vs-Samsung
temp scale shift), so at deployment (always masks=1) the model drifts toward DREAMT's low-Deep behavior.
Modality dropout + personalization + threshold recalibration reduce but do **not** remove the harm.

**FINAL VERDICT — how to "use the data we now collect":**
- **Alarm (wake decision):** keep it **HR + accel only** — deploy the causal-feature model (35.4% / 83.0%,
  the 24%→35% precision win). Robust, cohort-agnostic, no cross-device risk. Temp/HRV do NOT belong here.
- **Report / 4-stage view:** USE the real temp + HRV for **visualization** (skin-temp curve, HRV/resting-HR
  trend) and a **separate 4-stage report model** (stakes are low; it's not the wake decision). This is the
  honest, valuable, zero-risk way to consume the new sensors.
- Artifact `models/deploy_v3/` kept for the record but is NOT recommended for the alarm.

### DEPLOY retrain — cohort-prevalence rebalancing + stronger modality dropout (2026-08-14)
Attempt to let temp/HRV ride in the alarm WITHOUT harm: reweight so DREAMT's effective Deep prevalence
matches Walch's 12% (breaks "sensors on → low Deep"), modality dropout 0.35→0.5.

| Condition | Walch Deep precision | recall |
|---|---|---|
| masks=1 first attempt | 23.7% | 83.0% |
| **masks=1 rebalanced (this)** | **28.0%** | 83.0% |
| HR+accel only (B0) | **35.4%** | 83.0% |

→ Rebalancing removed the recall collapse (masks=1 raw recall 49%→89%) and recovered precision 23.7→28.0%,
but a **~7-pt precision gap vs HR+accel remains**. Five experiments (C1, C3, deploy, deploy-recal, deploy-rebal)
now all agree: **temp/HRV in the alarm = neutral-to-harmful; best alarm is HR+accel causal (35.4/83.0).**
Conclusive. Use temp/HRV in the report + 4-stage model, not the wake decision.

---

## 4-stage REPORT model (uses all 4 signals) — deliverable
`build_report_figures.py`: XGBoost, Walch+DREAMT, 47 feats (HR+accel+real temp+HRV time/freq/personalized),
grouped 5-fold. **Overall 68.3% accuracy, Cohen's κ = 0.51.** Per-stage P/R: Wake 84/79, Light 75/66,
Deep 37/54, REM 37/56. Figures: `~/Downloads/report_confusion_matrix.png`, `report_hypnogram.png`.
κ=0.51 is squarely in the published wearable 4-stage range (κ≈0.49–0.66; see citations).

---

## 3rd cohort ADDED — Wearanize+ OA (healthy, wrist E4, all 4 signals) — 2026-08-14
20 subjects / 9,620 epochs, 100% HRV coverage (HRV derived from BVP peak-detection), REAL skin temp,
HEALTHY stage mix (good N3 **and** REM). Downloaded from RDR WebDAV (open access, no DUA). Pooled training
is now **Walch + DREAMT + Wearanize+ = 120 subjects, 3 devices, 59,445 epochs.**

### The confound is BROKEN — temp/HRV can now go in the alarm with ~no harm

| Alarm (Walch cohort, recall 83%) | Deep precision |
|---|---|
| temp/HRV, 2 cohorts, first attempt | 23.7% |
| temp/HRV, 2 cohorts, rebalanced | 28.0% |
| **temp/HRV, 3 cohorts (+ healthy Wearanize+)** | **33.1%** |
| HR+accel only (B0) | 35.4% |

→ Adding a HEALTHY temp+HRV cohort means "sensors on" no longer implies "low Deep" (Wearanize+ is masks=1
AND high-deep). The on-watch (masks=1) deep precision jumped **23.7% → 33.1%**, now within ~2 pts of the
HR+accel baseline and equal to the masks=0 point (33.8%). **The recall collapse is gone.** So the model can
now consume all 4 signals in the alarm with negligible cost — the user's goal ("use the data, don't damage").

### 4-stage report model, 3 cohorts (uses all 4 signals)
Overall 66.0% acc, κ=0.49 (κ dipped from 0.51 = harder 3-device test set). Per-stage: Wake 81/77, Light
74/62, **Deep 42/62 (up from 37/54)**, REM 38/58 (flat — consistent with HRV-doesn't-crack-REM). The healthy
cohort improved **Deep** in the report; REM stays HR-driven. Figures refreshed in `~/Downloads`.

---

## Datasets to expand training (research Aug 14 2026)

| Dataset | Signals | Population | Deep | Access | Note |
|---|---|---|---|---|---|
| **Wearanize+** ⭐ | HR+accel+**temp+HRV** (wrist E4) | healthy | normal | open (Radboud click-through) | only DREAMT+this have skin temp; verify OA download has E4 streams |
| **BIDsleep** | HR+accel | healthy | normal | instant (PhysioNet) | multi-night (253), Apple Watch, easy volume |
| **MESA** | HR/HRV (ECG+PPG)+actigraphy | older, multi-ethnic | low | NSRR DUA (advisor sig) | no temp; NSRR has no body-temp anywhere |

Skip: MMASH (no PSG labels), BOAS (headband not wrist), WESAD/PPG-DaLiA (no sleep labels).

## Key citations (verified) — for the defense
**Method (pooling multi-cohort):** U-Sleep (Perslev 2021, npj Dig Med, 16 cohorts/15,660 subj), RobustSleepNet
(Guillot 2021, 8 datasets). **Missing signals:** missingness indicators (Little & Rubin 2019; sklearn),
XGBoost native NaN (Chen & Guestrin 2016), ModDrop (Neverova 2016), SMIL (Ma 2021). **Cross-device error:**
Bent 2020 (npj Dig Med, E4 vs Apple Watch). **Confirms HR carries the signal / HRV+temp marginal:**
Sridhar 2020 (HR-alone = 77%/κ0.66, matches HRV pipelines), Walch 2019 (HR = dominant gain), Kräuchi 1999
+ Kwon PNAS (skin temp helps sleep-ONSET, not staging). → our results reproduce the literature.
