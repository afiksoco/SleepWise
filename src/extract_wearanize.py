"""
Wearanize+ OA (Empatica E4, HEALTHY cohort) -> epoch features, the 3rd training
cohort. Same feature DEFINITIONS as DREAMT (reuses features.py + extract_dreamt_hrv
HRV functions) so the columns line up for pooled training.

Parquet layout: one row per device; the 'Empatica E4' row's SignalData is a dict of
channels — HR(1Hz), TEMP(4Hz), ACCX/Y/Z(32Hz, raw E4 1/64 g), BVP(64Hz PPG).
No IBI channel, so HRV is derived from BVP peak-detection. Labels: the PSG row's
SleepScores['ManualScores1'] (974 epochs @30s, AASM ints 0=W 1=N1 2=N2 3=N3 4=REM),
aggregated to 60 s. PlugNPlay is already synchronized + truncated, so signals and
scores share one timeline.

Output: wearanize_feats.pkl
"""
import os, sys, glob, time, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))
from features import (extract_hr_features, extract_temp_features, extract_acc_features,
                      add_temporal_features)
from extract_dreamt_hrv import hrv_for_epoch, hrv_freq
from scipy.signal import find_peaks

DATA = os.path.expanduser('~/PycharmProjects/SleepWise/data/wearanize')
CACHE = os.path.join(os.path.dirname(__file__), '..', 'wearanize_feats.pkl')
EPOCH = 60
AASM = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}


def rr_from_bvp(bvp, fs=64):
    """Systolic-peak detection on the E4 BVP (PPG) -> RR series (ms), gated."""
    x = np.asarray(bvp, float)
    x = x[~np.isnan(x)]
    if len(x) < fs * 5:
        return np.array([])
    x = x - np.mean(x)
    sd = np.std(x) or 1.0
    peaks, _ = find_peaks(x, distance=int(0.4 * fs), prominence=0.3 * sd)  # <=150 bpm
    if len(peaks) < 3:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000.0
    return rr[(rr >= 300) & (rr <= 2000)]


def extract_subject(df):
    e4 = df[df['Device'].astype(str).str.contains('mpatica', na=False)]
    psg = df[df['Device'].astype(str).str.contains('omno|Mentalab|PSG', na=False)]
    if len(e4) == 0 or len(psg) == 0:
        return None, None
    sd = e4.iloc[0]['SignalData']; sr = e4.iloc[0]['SamplingRate']
    def ch(name):
        k = next((k for k in sd if k.upper() == name), None)
        return (np.asarray(sd[k], float), float(sr[k])) if k and sd.get(k) is not None else (None, None)
    hr, f_hr = ch('HR'); temp, f_t = ch('TEMP'); bvp, f_b = ch('BVP')
    ax, f_a = ch('ACCX'); ay, _ = ch('ACCY'); az, _ = ch('ACCZ')
    if hr is None or temp is None or bvp is None or ax is None:
        return None, None
    # labels
    scores = None
    for _, r in psg.iterrows():
        s = r['SleepScores']
        if isinstance(s, dict) and 'ManualScores1' in s:
            scores = np.asarray(s['ManualScores1']); break
    if scores is None:
        return None, None
    n_ep = int(min(len(hr) // int(f_hr * EPOCH), len(scores) // 2))    # 60 s epochs
    rows, labs = [], []
    for i in range(n_ep):
        def win(sig, fs, sec0=0):
            a = int((i * EPOCH - sec0) * fs); b = int(((i + 1) * EPOCH) * fs)
            return sig[max(a, 0):b]
        feat = {}
        feat.update(extract_hr_features(win(hr, f_hr)))
        feat.update(extract_temp_features(win(temp, f_t)))
        feat.update(extract_acc_features(win(ax, f_a), win(ay, f_a), win(az, f_a)))
        rr_ep = rr_from_bvp(bvp[int(i * EPOCH * f_b):int((i + 1) * EPOCH * f_b)])
        rr_win = rr_from_bvp(bvp[int(max(0, i * EPOCH - 60) * f_b):int((i + 1) * EPOCH * f_b)])
        feat.update(hrv_for_epoch(rr_ep)); feat.update(hrv_freq(rr_win))
        rows.append(feat)
        lab2 = scores[i * 2:i * 2 + 2]
        vals = [AASM.get(int(v), 'W') for v in lab2 if not np.isnan(float(v))]
        labs.append(max(set(vals), key=vals.count) if vals else 'W')
    fdf = add_temporal_features(pd.DataFrame(rows), lookback=4)
    return fdf, np.array(labs)


def main():
    files = sorted(glob.glob(os.path.join(DATA, 'Sub*.parquet')))
    print(f"extracting {len(files)} Wearanize+ subjects...", flush=True)
    parts, pid, t0 = [], 0, time.time()
    for f in files:
        try:
            df = pd.read_parquet(f)
            fdf, lab = extract_subject(df)
            if fdf is not None and len(lab) > 10:
                fdf['label_raw'] = lab; fdf['participant_id'] = pid
                parts.append(fdf)
                cov = 100 * fdf['hrv_rmssd'].notna().mean()
                dist = {k: int((lab == k).sum()) for k in ['W', 'N1', 'N2', 'N3', 'R']}
                print(f"  [{pid:2d}] {os.path.basename(f)}: {len(lab)} ep  HRVcov={cov:.0f}%  {dist}  ({time.time()-t0:.0f}s)", flush=True)
                pid += 1
            else:
                print(f"  {os.path.basename(f)}: no E4/labels, skipped", flush=True)
        except Exception as e:
            print(f"  {os.path.basename(f)} ERROR {repr(e)[:120]}", flush=True)
    if not parts:
        print("no subjects extracted"); return
    allf = pd.concat(parts, ignore_index=True)
    allf.to_pickle(CACHE)
    print(f"\ncached {len(allf)} epochs, {pid} subjects -> {CACHE} ({time.time()-t0:.0f}s)")
    print("temp real?", allf['temp_mean'].nunique(), "uniq; HRV rows:", int(allf['hrv_rmssd'].notna().sum()))


if __name__ == '__main__':
    main()
