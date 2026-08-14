"""
Build DREAMT epoch features INCLUDING real HRV — the piece the old
extract_cache.py dropped ("HRV removed as DREAMT has no IBI").

DREAMT whole_df actually DOES carry IBI: the column is forward-filled at 64 Hz
(96% non-null, values in SECONDS). The true inter-beat intervals are the *value
changes*, so we dedupe consecutive repeats to recover the RR series per epoch and
compute standard time-domain HRV (RMSSD / SDNN / pNN50 / mean-IBI).

Reuses the shared extractor for HR (9) + temp (3) + accel (2) + temporal, so these
features are defined IDENTICALLY to Walch — then appends the 4 HRV columns.

Output: dreamt_feats.pkl  (one row per 60 s epoch; real temp + real HRV)
Env: MAX_SUBJECTS (default: all 69).
"""
import os, sys, glob, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features_from_participant

DREAMT = os.path.expanduser('~/PycharmProjects/SleepWise/data/dreamt')
CACHE = os.path.join(os.path.dirname(__file__), '..', 'dreamt_feats.pkl')
FS, EPOCH = 64, 60
SPE = FS * EPOCH                      # samples per epoch = 3840
MAX_SUBJECTS = int(os.environ.get('MAX_SUBJECTS', '69'))


def rr_from_ibi(ibi_slice_sec: np.ndarray) -> np.ndarray:
    """Recover the true RR beat series (ms) from forward-filled IBI (seconds):
    real beats = points where the value changes. Physiological gate 300-2000 ms."""
    x = ibi_slice_sec[~np.isnan(ibi_slice_sec)]
    if len(x) < 2:
        return np.array([])
    rr = x[np.concatenate(([True], np.diff(x) != 0))] * 1000.0
    return rr[(rr >= 300) & (rr <= 2000)]


def hrv_freq(rr: np.ndarray) -> dict:
    """Frequency-domain HRV over a (causal) RR window — the classic REM signal.
    Interpolate the RR tachogram to 4 Hz, Hann-window, FFT, integrate LF/HF bands.
    LF=0.04-0.15 Hz, HF=0.15-0.40 Hz. Log-power (stored) + LF/HF ratio."""
    if len(rr) < 8:
        return {'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lfhf': np.nan}
    t = np.cumsum(rr) / 1000.0; t -= t[0]
    if t[-1] < 20:                       # need ~20 s of beats for a stable estimate
        return {'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lfhf': np.nan}
    fs = 4.0
    tt = np.arange(0, t[-1], 1 / fs)
    if len(tt) < 16:
        return {'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lfhf': np.nan}
    rri = np.interp(tt, t, rr); rri = rri - rri.mean()
    psd = np.abs(np.fft.rfft(rri * np.hanning(len(rri)))) ** 2
    freq = np.fft.rfftfreq(len(rri), 1 / fs)
    lf = psd[(freq >= 0.04) & (freq < 0.15)].sum()
    hf = psd[(freq >= 0.15) & (freq < 0.40)].sum()
    return {'hrv_lf': float(np.log1p(lf)), 'hrv_hf': float(np.log1p(hf)),
            'hrv_lfhf': float(lf / hf) if hf > 0 else np.nan}


def hrv_for_epoch(rr: np.ndarray) -> dict:
    """Time-domain HRV from the epoch's RR series (ms)."""
    if len(rr) < 5:
        return {'hrv_rmssd': np.nan, 'hrv_sdnn': np.nan, 'hrv_pnn50': np.nan, 'hrv_ibi_mean': np.nan}
    dif = np.diff(rr)
    return {
        'hrv_rmssd': float(np.sqrt(np.mean(dif ** 2))),
        'hrv_sdnn': float(np.std(rr)),
        'hrv_pnn50': float(np.mean(np.abs(dif) > 50.0) * 100.0),
        'hrv_ibi_mean': float(np.mean(rr)),
    }


def main():
    files = sorted(glob.glob(os.path.join(DREAMT, 'S*_whole_df.csv')))[:MAX_SUBJECTS]
    print(f"extracting {len(files)} DREAMT subjects (HR+temp+accel+HRV)...", flush=True)
    parts, pid, t0 = [], 0, time.time()
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['HR', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'IBI', 'Sleep_Stage'])
            fdf, lab = extract_features_from_participant(df, add_temporal=True)   # HR/temp/accel + temporal
            # HRV per epoch over the SAME index windows the extractor used.
            # Time-domain: current 60 s epoch. Freq-domain: causal 120 s window
            # ending at the epoch (needs more beats for a stable LF/HF estimate).
            n = len(lab)
            ibi = df['IBI'].values
            hrv_rows = []
            for i in range(n):
                rr_ep = rr_from_ibi(ibi[i * SPE:(i + 1) * SPE])
                rr_win = rr_from_ibi(ibi[max(0, i * SPE - 60 * FS):(i + 1) * SPE])
                row = hrv_for_epoch(rr_ep); row.update(hrv_freq(rr_win))
                hrv_rows.append(row)
            hrv_df = pd.DataFrame(hrv_rows)
            fdf = pd.concat([fdf.reset_index(drop=True), hrv_df], axis=1)
            if n > 10:
                fdf['label_raw'] = lab
                fdf['participant_id'] = pid
                parts.append(fdf)
                cov = 100 * hrv_df['hrv_rmssd'].notna().mean()
                print(f"  [{pid:2d}] {os.path.basename(f)}: {n} epochs  HRV-cov={cov:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
                pid += 1
        except Exception as e:
            print(f"  {os.path.basename(f)} ERROR {e}", flush=True)
    allf = pd.concat(parts, ignore_index=True)
    allf.to_pickle(CACHE)
    print(f"\ncached {len(allf)} epochs x {allf.shape[1]} cols, {pid} subjects -> {CACHE} ({time.time()-t0:.0f}s)", flush=True)
    # sanity
    print("temp_mean real?", allf['temp_mean'].nunique(), "unique;",
          "HRV rows:", int(allf['hrv_rmssd'].notna().sum()))


if __name__ == '__main__':
    main()
