"""
Feature-parity check: does the ON-DEVICE Kotlin feature builder
(TFLiteSleepPredictor.computeFeatureVector) reproduce the Python training
pipeline's 45-feature vector exactly?

We re-implement the Kotlin streaming logic here in Python (mirroring the Kotlin
line-for-line) and compare it, per feature per epoch, against the real pipeline
output (build_deploy_model.FEATS after load()/add_causal). A near-zero max-abs
diff for epochs >= WARMUP means the on-device ALGORITHM is correct.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from build_combined_model import add_causal, FEATS_C3  # noqa
from build_deploy_model import FEATS                    # the 45 deployed features

WARMUP = 5


def reference_vectors(sub):
    """Pipeline truth: add_causal (which the model was trained on) + masks."""
    d = add_causal(sub.copy())
    d['has_temp'] = d['temp_mean'].notna().astype(float)
    d['has_hrv'] = d['hrv_rmssd'].notna().astype(float)
    return d[FEATS].astype('float32').values


def kotlin_sim(sub):
    """Mirror of TFLiteSleepPredictor.computeFeatureVector, epoch-streamed."""
    hr = sub['hr_mean'].values.astype(float)
    acc = sub['acc_std'].values.astype(float)
    temp = sub['temp_mean'].values.astype(float)
    rmssd = sub['hrv_rmssd'].values.astype(float)
    ibim = sub['hrv_ibi_mean'].values.astype(float)
    base = {c: sub[c].values.astype(float) for c in
            ['hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
             'acc_std','acc_move_ratio','temp_mean','temp_std','temp_trend',
             'hrv_rmssd','hrv_sdnn','hrv_pnn50','hrv_ibi_mean']}

    def std_sample(xs):
        xs = [x for x in xs]
        if len(xs) < 2: return 0.0
        m = np.mean(xs); return float(np.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)))
    def mean(xs): return float(np.mean(xs)) if len(xs) else 0.0
    def expZ(hist):
        cur = hist[-1]
        if cur is None or (isinstance(cur,float) and np.isnan(cur)): return np.nan
        vals = [v for v in hist if v is not None and not (isinstance(v,float) and np.isnan(v))]
        m = np.mean(vals); s = 1.0 if len(vals) < 2 else std_sample(vals)
        return (cur - m) / (s + 1e-6)

    out = []
    for i in range(len(hr)):
        f = {}
        for c in base: f[c] = base[c][i]
        H = hr[:i+1]
        lag = lambda k: hr[i-k] if i-k >= 0 else hr[i]
        last4 = list(hr[max(0,i-3):i+1])
        tb = hr[i-4] if i-4 >= 0 else None
        f['hr_mean_lag1']=lag(1); f['hr_mean_lag2']=lag(2); f['hr_mean_lag3']=lag(3); f['hr_mean_lag4']=lag(4)
        f['hr_mean_rolling_mean']=mean(last4); f['hr_mean_rolling_std']=std_sample(last4)
        f['hr_mean_trend']=(hr[i]-tb) if tb is not None else 0.0
        f['hr_mean_roc']=((hr[i]-tb)/tb) if (tb is not None and tb!=0) else 0.0
        f['hr_stability']=std_sample(last4)
        f['sleep_cycle_position']=(i%90)/90.0
        f['min_since_onset']=float(i)
        f['hr_z']=expZ([float(x) for x in H])
        f['hr_drop_from_max']=float(np.max(H))-hr[i]
        for w in (5,10,15):
            f[f'hr_rmean{w}']=mean(list(hr[max(0,i-w+1):i+1]))
            f[f'hr_rstd{w}']=std_sample(list(hr[max(0,i-w+1):i+1]))
            f[f'acc_rmean{w}']=mean(list(acc[max(0,i-w+1):i+1]))
        f['temp_mean_z']=expZ(list(temp[:i+1]))
        f['hrv_rmssd_z']=expZ(list(rmssd[:i+1]))
        f['hrv_ibi_mean_z']=expZ(list(ibim[:i+1]))
        f['has_temp']=0.0 if np.isnan(temp[i]) else 1.0
        f['has_hrv']=0.0 if np.isnan(rmssd[i]) else 1.0
        out.append([f[name] for name in FEATS])
    return np.array(out, dtype='float32')


def main():
    # a Wearanize subject exercises temp+HRV; a DREAMT subject too. Use both.
    for pkl, label in [('wearanize_feats.pkl','Wearanize'), ('dreamt_feats.pkl','DREAMT')]:
        df = add_causal.__self__ if False else pd.read_pickle(os.path.join(os.path.dirname(__file__),'..',pkl))
        pid = df['participant_id'].value_counts().idxmax()
        sub = df[df['participant_id']==pid].reset_index(drop=True)
        ref = reference_vectors(sub); sim = kotlin_sim(sub)
        d = np.abs(np.nan_to_num(ref) - np.nan_to_num(sim))[WARMUP:]
        worst = sorted(((FEATS[j], float(d[:,j].max())) for j in range(len(FEATS))), key=lambda t:-t[1])
        print(f"\n=== {label} subj {pid}: {sub.shape[0]} epochs, {len(FEATS)} features ===")
        print(f"overall max|diff| (epochs>={WARMUP}) = {d.max():.6g}")
        print("worst 6 features:", [(n, round(v,6)) for n,v in worst[:6]])
        nan_mismatch = (np.isnan(ref)!=np.isnan(sim))[WARMUP:].sum()
        print(f"NaN-pattern mismatches: {nan_mismatch}")


if __name__ == '__main__':
    main()
