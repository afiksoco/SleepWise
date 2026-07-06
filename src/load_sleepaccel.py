"""
Loader for the Walch 2019 sleep-accel dataset (Apple Watch HR+accel + PSG labels).

Produces the SAME 32-feature contract the Android app uses (ANDROID_ORDER), so a
model trained on this is a drop-in .tflite swap. Key differences vs the DREAMT
loader, handled here:
  - accel is ALREADY in g (no /64 E4 conversion). Gravity removed via per-epoch
    median magnitude, same as the Android side.
  - HR is sparse PPG bpm (every few sec), not dense 64 Hz.
  - skin temp does not exist -> injected as the inference constant (34/0/0) so
    add_temporal_features emits the exact constant temp block the watch feeds.
  - labels are 30 s PSG epochs (wake=0,N1=1,N2=2,N3=3,REM=5,-1=unscored),
    aggregated to 60 s epochs by majority vote.
  - Apple-Watch motion coverage is partial (~30% of epochs); missing accel is
    returned as NaN (NOT 0, which would falsely signal "still -> deep") and
    imputed from neighbours by add_temporal_features' bfill/ffill.

Download from PhysioNet (see README) into
  data/sleep-accel-dl/physionet.org/files/sleep-accel/1.0.0/{motion,heart_rate,labels}/
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_hr_features, add_temporal_features, simplify_labels

DL = os.path.expanduser('~/PycharmProjects/SleepWise/data/sleep-accel-dl/physionet.org/files/sleep-accel/1.0.0')
EPOCH = 60           # seconds, to match Android's 1-min epochs
MOVE_THRESH_G = 0.02

ANDROID_ORDER = [
    'hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
    'temp_mean','temp_std','temp_trend',
    'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
    'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
    'temp_mean_lag1','temp_mean_lag2','temp_mean_lag3','temp_mean_lag4',
    'temp_mean_rolling_mean','temp_mean_rolling_std','temp_mean_trend','temp_mean_roc',
    'hr_stability','sleep_cycle_position',
    'acc_std','acc_move_ratio',
]
STAGE_MAP = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'R'}   # -1 dropped


def _acc_features_g(x, y, z):
    """Accel features for data ALREADY in g (Apple Watch / Galaxy Watch).

    Returns NaN when a window has too few samples — acc_std=0 is a strong
    "perfectly still -> deep" cue and must not be fabricated for missing data.
    """
    mag = np.sqrt(x * x + y * y + z * z)
    mag = mag[~np.isnan(mag)]
    if len(mag) < 5:
        return np.nan, np.nan
    motion = np.abs(mag - np.median(mag))            # gravity + posture removed
    return float(np.std(motion)), float(np.mean(motion > MOVE_THRESH_G))


def load_subject(sid, pid):
    lab = np.loadtxt(f'{DL}/labels/{sid}_labeled_sleep.txt')
    if lab.ndim == 1:
        lab = lab.reshape(1, -1)
    hr = np.loadtxt(f'{DL}/heart_rate/{sid}_heartrate.txt', delimiter=',')
    mot = pd.read_csv(f'{DL}/motion/{sid}_acceleration.txt', sep=r'\s+',
                      header=None, names=['t', 'x', 'y', 'z']).values

    lab = lab[np.isin(lab[:, 1], list(STAGE_MAP.keys()))]
    if len(lab) < 20:
        return None
    t0, t1 = lab[:, 0].min(), lab[:, 0].max() + 30

    hr = hr[np.argsort(hr[:, 0])]
    mot = mot[np.argsort(mot[:, 0])]
    hr = hr[(hr[:, 0] >= t0 - EPOCH) & (hr[:, 0] < t1 + EPOCH)]
    mot = mot[(mot[:, 0] >= t0 - EPOCH) & (mot[:, 0] < t1 + EPOCH)]
    hrt, motx = hr[:, 0], mot[:, 0]

    rows, labels = [], []
    for e in range(int((t1 - t0) // EPOCH)):
        s, ecol = t0 + e * EPOCH, t0 + e * EPOCH + EPOCH
        m = lab[(lab[:, 0] >= s) & (lab[:, 0] < ecol)]
        if len(m) == 0:
            continue
        stages, cnts = np.unique(m[:, 1], return_counts=True)
        raw = STAGE_MAP[int(stages[cnts.argmax()])]

        i0, i1 = np.searchsorted(hrt, s), np.searchsorted(hrt, ecol)
        feat = extract_hr_features(hr[i0:i1, 1])
        j0, j1 = np.searchsorted(motx, s), np.searchsorted(motx, ecol)
        astd, amove = _acc_features_g(mot[j0:j1, 1], mot[j0:j1, 2], mot[j0:j1, 3])
        feat['acc_std'] = astd
        feat['acc_move_ratio'] = amove
        feat['temp_mean'] = 34.0     # watch has no live skin temp -> inference constant
        feat['temp_std'] = 0.0
        feat['temp_trend'] = 0.0
        rows.append(feat)
        labels.append(raw)

    if len(rows) < 20:
        return None
    fdf = add_temporal_features(pd.DataFrame(rows), lookback=4)  # imputes NaN accel via bfill/ffill
    fdf = fdf.reindex(columns=ANDROID_ORDER).fillna(0)
    fdf['label_raw'] = labels
    fdf['participant_id'] = pid
    return fdf


def build_cache(out_pkl, max_subjects=None):
    sids = sorted(os.path.basename(f).split('_')[0]
                  for f in glob.glob(f'{DL}/labels/*_labeled_sleep.txt'))
    if max_subjects:
        sids = sids[:max_subjects]
    parts, pid = [], 0
    for sid in sids:
        if not (os.path.exists(f'{DL}/motion/{sid}_acceleration.txt')
                and os.path.exists(f'{DL}/heart_rate/{sid}_heartrate.txt')):
            continue
        try:
            r = load_subject(sid, pid)
            if r is not None:
                nd = (simplify_labels(r['label_raw'].values, 'binary_n3') == 'Deep').sum()
                print(f"  [{pid:2d}] {sid}: {len(r)} epochs, {nd} Deep ({100*nd/len(r):.1f}%)", flush=True)
                parts.append(r); pid += 1
        except Exception as ex:
            print(f"  [err] {sid}: {ex}", flush=True)
    allf = pd.concat(parts, ignore_index=True)
    allf.to_pickle(out_pkl)
    print(f"\ncached {len(allf)} epochs x {allf.shape[1]} cols, {pid} subjects -> {out_pkl}", flush=True)
    return allf


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'sleepaccel_feats.pkl')
    build_cache(out, int(sys.argv[1]) if len(sys.argv) > 1 else None)
