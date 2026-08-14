"""
The "our model uses all four signals" deliverable — a 4-stage sleep-report model
(Wake/Light/Deep/REM) trained on Walch + DREAMT with HR + accel + REAL temp + HRV
(time + frequency + personalized). This is the REPORT model, separate from the
HR+accel alarm — here temp/HRV are safe to use and genuinely contribute.

Produces presentation assets:
  ~/Downloads/report_confusion_matrix.png   (row-normalized 4x4, English labels)
  ~/Downloads/report_hypnogram.png          (one night: predicted vs ground truth)
and prints per-class precision/recall + overall accuracy + Cohen's kappa.
"""
import os, sys, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))
from build_combined_model import load, FEATS_C3
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.expanduser('~/Downloads')
CLASSES = ['Wake', 'Light', 'Deep', 'REM']


def main():
    df = load()
    lab = simplify_labels(df['label_raw'].values, '4class')
    ok = np.isin(lab, CLASSES)
    df = df[ok].reset_index(drop=True); lab = lab[ok]
    cmap = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([cmap[c] for c in lab]); g = df['participant_id'].values; coh = df['cohort'].values
    X = df[FEATS_C3].astype('float32').values
    print(f"4-stage report model: {len(y)} epochs, {len(FEATS_C3)} features (HR+accel+temp+HRV)")

    oof = np.zeros((len(y), 4))
    for tr, te in GroupKFold(5).split(X, y, g):
        w = np.array([len(y[tr]) / (4 * max((y[tr] == c).sum(), 1)) for c in y[tr]])
        m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, objective='multi:softprob', num_class=4,
                              eval_metric='mlogloss', n_jobs=-1, missing=np.nan)
        m.fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict_proba(X[te])
    pred = oof.argmax(1)

    # ---- metrics ----
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    acc = (pred == y).mean(); kappa = cohen_kappa_score(y, pred)
    print(f"\n{'stage':<8}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>8}")
    for i, c in enumerate(CLASSES):
        print(f"{c:<8}{p[i]*100:>6.0f}%{r[i]*100:>6.0f}%{f[i]:>7.2f}{int(s[i]):>8}")
    print(f"\noverall accuracy={acc*100:.1f}%   Cohen's kappa={kappa:.3f}")

    # ---- confusion matrix (row-normalized) ----
    cm = confusion_matrix(y, pred, labels=range(4)).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True (PSG)')
    ax.set_title(f"4-stage sleep report — combined model\naccuracy {acc*100:.0f}%, kappa {kappa:.2f}")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}%", ha='center', va='center',
                    color='white' if cmn[i, j] > 0.5 else 'black', fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/report_confusion_matrix.png", dpi=140); plt.close(fig)
    print(f"saved {OUT}/report_confusion_matrix.png")

    # ---- hypnogram: pick the DREAMT subject with the most epochs (has all 4 signals) ----
    dmask = coh == 'dreamt'
    subj = pd.Series(g[dmask]).value_counts().idxmax()
    idx = np.where(g == subj)[0]
    order = {0: 3, 3: 2, 1: 1, 2: 0}           # plot order: Wake(top) REM Light Deep(bottom)
    yt = np.array([order[v] for v in y[idx]]); yp = np.array([order[v] for v in pred[idx]])
    t = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.step(t, yt, where='mid', color='#222', lw=1.6, label='Ground truth (PSG)')
    ax.step(t, yp, where='mid', color='#d1495b', lw=1.2, alpha=0.8, label='Model (4-stage)')
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(['Wake', 'REM', 'Light', 'Deep'])
    ax.set_xlabel('Time (minutes)'); ax.set_title('Full-night hypnogram — model uses HR + accel + skin-temp + HRV')
    ax.legend(loc='upper right', fontsize=9); ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(f"{OUT}/report_hypnogram.png", dpi=140); plt.close(fig)
    print(f"saved {OUT}/report_hypnogram.png  (subject {subj}, {len(idx)} min)")


if __name__ == '__main__':
    main()
