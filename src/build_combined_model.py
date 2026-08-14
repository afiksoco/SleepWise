"""
Stage 3: ONE model on pooled Walch + DREAMT using all four signals.

HR + accel are present in both cohorts; temp + HRV only in DREAMT (Empatica E4)
and, at inference, on the Samsung watch. Missing temp/HRV are handled as
missingness (NaN for XGBoost; zeroed + presence-mask inputs for the NN) —
standard multi-cohort / missing-modality practice (cf. U-Sleep).

Everything is grouped 5-fold by SUBJECT across both cohorts (no leakage), and
every result is reported PER-COHORT as well as pooled, so DREAMT's low Deep
prevalence can't silently distort the Walch-comparable numbers.

Experiments:
  A. Binary Deep/Light  — compare Walch-cohort to baseline B0 (Deep P .354 / R .830).
  B. 4-class (Wake/Light/Deep/REM) ABLATION: HR+accel only  vs  +temp+HRV.
     This is the money test — does HRV actually make REM detectable?
Also: XGBoost feature importances (is temp/HRV even used?).
"""
import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb

HERE = os.path.dirname(__file__)
R = lambda p: os.path.join(HERE, '..', p)

# --- feature groups -----------------------------------------------------------
BASE = ['hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
        'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
        'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
        'hr_stability','sleep_cycle_position','acc_std','acc_move_ratio']
NEW = ['min_since_onset','hr_z','hr_drop_from_max'] + [f'hr_rmean{w}' for w in (5,10,15)] + \
      [f'hr_rstd{w}' for w in (5,10,15)] + [f'acc_rmean{w}' for w in (5,10,15)]
HR_ACCEL = BASE + NEW                       # present in BOTH cohorts
TEMP = ['temp_mean','temp_std','temp_trend']            # DREAMT + watch only
HRV_TIME = ['hrv_rmssd','hrv_sdnn','hrv_pnn50','hrv_ibi_mean']   # C1
HRV_FREQ = ['hrv_lf','hrv_hf','hrv_lfhf']                        # C3: LF/HF frequency-domain
PERS = ['hrv_rmssd_z','hrv_ibi_mean_z','hrv_lfhf_z','temp_mean_z']  # C3: personalized (per-subject causal z)
HRV = HRV_TIME                              # back-compat alias
# C1 feature set = time-domain HRV only; C3 = + frequency-domain + personalized baselining
FEATS_C1 = HR_ACCEL + TEMP + HRV_TIME
FEATS_C3 = HR_ACCEL + TEMP + HRV_TIME + HRV_FREQ + PERS

# columns that get a personalized (per-subject, causal/expanding) z-score
_PERS_SRC = {'hrv_rmssd': 'hrv_rmssd_z', 'hrv_ibi_mean': 'hrv_ibi_mean_z',
             'hrv_lfhf': 'hrv_lfhf_z', 'temp_mean': 'temp_mean_z'}


def add_causal(d):
    out = []
    for _, sub in d.groupby('participant_id', sort=False):
        sub = sub.copy().reset_index(drop=True)
        sub['min_since_onset'] = np.arange(len(sub))
        em = sub['hr_mean'].expanding().mean(); es = sub['hr_mean'].expanding().std().fillna(1) + 1e-6
        sub['hr_z'] = (sub['hr_mean'] - em) / es
        sub['hr_drop_from_max'] = sub['hr_mean'].cummax() - sub['hr_mean']
        for w in (5, 10, 15):
            sub[f'hr_rmean{w}'] = sub['hr_mean'].rolling(w, min_periods=1).mean()
            sub[f'hr_rstd{w}']  = sub['hr_mean'].rolling(w, min_periods=1).std().fillna(0)
            sub[f'acc_rmean{w}'] = sub['acc_std'].rolling(w, min_periods=1).mean()
        # personalized causal z-scores (absolute HRV/temp vary hugely between people;
        # the model needs each subject's OWN deviation, like hr_z). NaN stays NaN (Walch).
        for src, dst in _PERS_SRC.items():
            if src in sub:
                m = sub[src].expanding().mean(); s = sub[src].expanding().std().fillna(1) + 1e-6
                sub[dst] = (sub[src] - m) / s
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def load():
    W = add_causal(pd.read_pickle(R('sleepaccel_feats.pkl')))
    for c in TEMP + HRV_TIME + HRV_FREQ + PERS:
        W[c] = np.nan                            # Walch: temp was a fake constant; no HRV
    W['cohort'] = 'walch'; W['participant_id'] = W['participant_id'] + 1000

    D = add_causal(pd.read_pickle(R('dreamt_feats.pkl')))
    D['cohort'] = 'dreamt'

    frames = [D, W]
    # 3rd cohort: Wearanize+ OA (healthy, wrist E4, all 4 signals) — optional if extracted
    wz_path = R('wearanize_feats.pkl')
    if os.path.exists(wz_path):
        Z = add_causal(pd.read_pickle(wz_path))
        Z['cohort'] = 'wearanize'; Z['participant_id'] = Z['participant_id'] + 2000
        frames.append(Z)

    df = pd.concat(frames, ignore_index=True)
    for c in FEATS_C3:                           # guarantee every feature column exists
        if c not in df: df[c] = np.nan
    return df


def cohort_metrics(y, pred, cohort, classes):
    """Per-class precision/recall for pooled and each cohort."""
    rows = {}
    for name, mask in [('pooled', np.ones(len(y), bool)),
                       ('walch', cohort == 'walch'), ('dreamt', cohort == 'dreamt'),
                       ('wearanize', cohort == 'wearanize')]:
        if mask.sum() == 0:
            continue
        p, r, f, s = precision_recall_fscore_support(y[mask], pred[mask], labels=classes,
                                                     average=None, zero_division=0)
        rows[name] = {c: (p[i], r[i], f[i], int(s[i])) for i, c in enumerate(classes)}
    return rows


# ============================ A. BINARY DEEP/LIGHT ============================
def exp_binary(df):
    print("\n" + "=" * 78 + "\nA. BINARY Deep/Light (Deep=N3) — combined, grouped CV, per-cohort\n" + "=" * 78)
    y = (simplify_labels(df['label_raw'].values, 'binary_n3') == 'Deep').astype(int)
    g = df['participant_id'].values
    coh = df['cohort'].values
    FEATS = FEATS_C3
    X = df[FEATS].astype('float32').values          # NaN kept for XGBoost
    print(f"n={len(y)}  Deep pooled={y.mean()*100:.1f}%  "
          f"(walch={y[coh=='walch'].mean()*100:.1f}%  dreamt={y[coh=='dreamt'].mean()*100:.1f}%)")

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              eval_metric='logloss', n_jobs=-1, missing=np.nan)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    # threshold calibrated to recall ~0.83 on the WALCH cohort (apples-to-apples with B0)
    wm = coh == 'walch'
    p, r, th = precision_recall_curve(y[wm], oof[wm])
    i = int(np.argmin(np.abs(r - 0.83))); thr = th[min(i, len(th) - 1)]
    pred = (oof >= thr).astype(int)
    print(f"\noperating threshold (walch recall~0.83): {thr:.3f}")
    mets = cohort_metrics(y, pred, coh, [1])       # class 1 = Deep
    print(f"{'cohort':<8}{'Deep prec':>11}{'Deep rec':>10}{'Deep F1':>9}{'n_deep':>9}")
    for name, d in mets.items():
        pr, rc, f1, s = d[1]
        print(f"{name:<8}{pr*100:>10.1f}%{rc*100:>9.1f}%{f1:>9.3f}{s:>9}")
    print("baseline B0 (walch-only NN): Deep prec 35.4%  rec 83.0%  F1 0.496")
    return {'thr': float(thr), 'metrics': {k: {'deep': v[1]} for k, v in mets.items()}}


# ============================ B. 4-CLASS + REM ABLATION ======================
def exp_4class(df):
    print("\n" + "=" * 78 + "\nB. 4-CLASS (Wake/Light/Deep/REM) — ABLATION: does temp+HRV enable REM?\n" + "=" * 78)
    lab = simplify_labels(df['label_raw'].values, '4class')
    classes = ['Wake', 'Light', 'Deep', 'REM']
    cmap = {c: i for i, c in enumerate(classes)}
    ok = np.isin(lab, classes)
    df = df[ok].reset_index(drop=True); lab = lab[ok]
    y = np.array([cmap[c] for c in lab]); g = df['participant_id'].values; coh = df['cohort'].values
    print("class counts:", {c: int((y == cmap[c]).sum()) for c in classes})
    print(f"REM only exists where labeled — walch REM={int((y[coh=='walch']==3).sum())}, "
          f"dreamt REM={int((y[coh=='dreamt']==3).sum())}")

    results = {}
    for tag, feats in [('HR+accel only', HR_ACCEL), ('+temp+HRV+LF/HF+pers (C3)', FEATS_C3)]:
        X = df[feats].astype('float32').values
        oof = np.zeros((len(y), 4))
        for tr, te in GroupKFold(5).split(X, y, g):
            w = np.array([ (y[tr]==0).sum()==0 and 1 or len(y[tr])/(4*max((y[tr]==c).sum(),1)) for c in y[tr] ])
            m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                                  num_class=4, eval_metric='mlogloss', n_jobs=-1, missing=np.nan)
            m.fit(X[tr], y[tr], sample_weight=w)
            oof[te] = m.predict_proba(X[te])
        pred = oof.argmax(1)
        mets = cohort_metrics(y, pred, coh, list(range(4)))
        print(f"\n--- {tag} ({len(feats)} feats) ---")
        print(f"{'cohort':<8}" + "".join(f"{c+' P/R':>14}" for c in classes))
        for name, d in mets.items():
            cells = "".join(f"{d[i][0]*100:>6.0f}/{d[i][1]*100:<6.0f}" for i in range(4))
            print(f"{name:<8}{cells}")
        # REM highlight (dreamt cohort — that's where REM+HRV coexist)
        rem = mets.get('dreamt', {}).get(3)
        if rem: print(f"    >> DREAMT REM: precision={rem[0]*100:.1f}%  recall={rem[1]*100:.1f}%  (n={rem[3]})")
        results[tag] = mets
    return results


def feature_importance(df):
    print("\n" + "=" * 78 + "\nFeature importance (binary Deep, gain) — is temp/HRV even used?\n" + "=" * 78)
    y = (simplify_labels(df['label_raw'].values, 'binary_n3') == 'Deep').astype(int)
    FEATS = FEATS_C3
    ALL_HRV = HRV_TIME + HRV_FREQ + [c for c in PERS if c.startswith('hrv')]
    X = df[FEATS].astype('float32').values
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, importance_type='gain',
                          n_jobs=-1, missing=np.nan).fit(X, y)
    imp = sorted(zip(FEATS, m.feature_importances_), key=lambda t: -t[1])
    for f, v in imp[:14]:
        tag = '  <-- TEMP' if f in TEMP or f == 'temp_mean_z' else ('  <-- HRV' if f in ALL_HRV else '')
        print(f"  {f:<22}{v:.4f}{tag}")
    tv = sum(v for f, v in imp if f in TEMP or f == 'temp_mean_z'); hv = sum(v for f, v in imp if f in ALL_HRV)
    print(f"  TOTAL temp gain share: {tv*100:.1f}%   HRV gain share: {hv*100:.1f}%")


def main():
    t0 = time.time()
    df = load()
    print(f"loaded combined: {len(df)} epochs  "
          f"(walch={int((df['cohort']=='walch').sum())}, dreamt={int((df['cohort']=='dreamt').sum())})  "
          f"subjects={df['participant_id'].nunique()}")
    print(f"HRV coverage: {df['hrv_rmssd'].notna().mean()*100:.1f}% of all rows "
          f"({df.loc[df.cohort=='dreamt','hrv_rmssd'].notna().mean()*100:.1f}% of DREAMT)")
    a = exp_binary(df)
    b = exp_4class(df)
    feature_importance(df)
    json.dump({'binary': a}, open(R('models/combined_results.json'), 'w'), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
