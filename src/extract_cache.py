"""Extract DREAMT features once and cache to a pickle (expensive: ~134MB/subject).

DREAMT = Empatica E4 wristband + PSG labels (sleep-disorder population).
Download from PhysioNet (see README) into data/dreamt/S*_whole_df.csv.
"""
import os, sys, glob, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features_from_participant

DREAMT = os.path.expanduser('~/PycharmProjects/SleepWise/data/dreamt')
CACHE = os.path.join(os.path.dirname(__file__), '..', 'dreamt_feats.pkl')
MAX_SUBJECTS = int(os.environ.get('MAX_SUBJECTS', '30'))

files = sorted(glob.glob(os.path.join(DREAMT, 'S*_whole_df.csv')))[:MAX_SUBJECTS]
print(f"extracting {len(files)} subjects...", flush=True)
parts, pid, t0 = [], 0, time.time()
for f in files:
    try:
        df = pd.read_csv(f, usecols=['HR', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Sleep_Stage'])
        fdf, lab = extract_features_from_participant(df, add_temporal=True)
        if len(lab) > 10:
            fdf['label_raw'] = lab
            fdf['participant_id'] = pid
            parts.append(fdf)
            print(f"  [{pid:2d}] {os.path.basename(f)}: {len(lab)} epochs  ({time.time()-t0:.0f}s)", flush=True)
            pid += 1
    except Exception as e:
        print(f"  {os.path.basename(f)} ERROR {e}", flush=True)

allf = pd.concat(parts, ignore_index=True)
allf.to_pickle(CACHE)
print(f"\ncached {len(allf)} epochs x {allf.shape[1]} cols -> {CACHE} ({time.time()-t0:.0f}s)", flush=True)
