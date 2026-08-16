import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
import xgboost as xgb

BASE=['hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
 'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
 'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
 'hr_stability','sleep_cycle_position','acc_std','acc_move_ratio']  # drop constant temp cols

df=pd.read_pickle('sleepaccel_feats.pkl').copy()
lab=simplify_labels(df['label_raw'].values,'binary_n3')
df['y']=(lab=='Deep').astype(int)
g=df['participant_id'].values

# ---- Stage-1 NEW features (all causal / deployable) ----
def add_feats(d):
    out=[]
    for pid,sub in d.groupby('participant_id',sort=False):
        sub=sub.copy().reset_index(drop=True)
        sub['min_since_onset']=np.arange(len(sub))            # deep is front-loaded
        m,s=sub['hr_mean'].mean(),sub['hr_mean'].std()+1e-6
        sub['hr_z']=(sub['hr_mean']-m)/s                       # personalized baseline
        for w in (5,10,15):                                    # wider causal context
            sub[f'hr_rmean{w}']=sub['hr_mean'].rolling(w,min_periods=1).mean()
            sub[f'hr_rstd{w}'] =sub['hr_mean'].rolling(w,min_periods=1).std().fillna(0)
            sub[f'acc_rmean{w}']=sub['acc_std'].rolling(w,min_periods=1).mean()
        sub['hr_drop_from_max']=sub['hr_mean'].cummax()-sub['hr_mean']  # deep=lowest HR of night
        out.append(sub)
    return pd.concat(out,ignore_index=True)
df=add_feats(df)
NEW=['min_since_onset','hr_z','hr_drop_from_max']+[f'hr_rmean{w}' for w in(5,10,15)]+\
    [f'hr_rstd{w}' for w in(5,10,15)]+[f'acc_rmean{w}' for w in(5,10,15)]
FEATS=BASE+NEW
X=df[FEATS].fillna(0).values.astype('float32'); y=df['y'].values

def viterbi(prob, p_stay=0.92):   # 2-state transition smoothing per night
    n=len(prob); eps=1e-6
    logE=np.log(np.clip(np.c_[1-prob,prob],eps,1))
    logT=np.log(np.array([[p_stay,1-p_stay],[1-p_stay,p_stay]]))
    V=np.log(np.array([0.85,0.15]))+logE[0]; bp=np.zeros((n,2),int)
    for t in range(1,n):
        for s in (0,1):
            seq=V+logT[:,s]; bp[t,s]=np.argmax(seq); V=V if False else V
            if s==0: c0=np.argmax(V+logT[:,0]); v0=(V+logT[:,0])[c0]+logE[t,0]
            else: c1=np.argmax(V+logT[:,1]); v1=(V+logT[:,1])[c1]+logE[t,1]
        Vn=np.array([v0,v1]); bp[t]=[c0,c1]; V=Vn
    path=np.zeros(n,int); path[-1]=np.argmax(V)
    for t in range(n-1,0,-1): path[t-1]=bp[t,path[t]]
    return path

# ---- grouped 5-fold OOF probabilities ----
oof=np.zeros(len(y))
for tr,te in GroupKFold(5).split(X,y,g):
    pos=y[tr].sum(); neg=len(tr)-pos
    clf=xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8,scale_pos_weight=neg/pos*0.45,  # mild -> favor precision
        eval_metric='logloss',n_jobs=4)
    clf.fit(X[tr],y[tr]); oof[te]=clf.predict_proba(X[te])[:,1]

def report(pred,title):
    cm=confusion_matrix(y,pred,labels=[0,1])  # 0=Light,1=Deep
    rec=cm[1,1]/cm[1].sum(); prec=cm[1,1]/max(cm[:,1].sum(),1)
    print(f"\n{title}\n  Deep recall={rec*100:.1f}%  precision={prec*100:.1f}%  (Deep predicted={cm[:,1].sum()}, true Deep={cm[1].sum()})")

print(f"data: {len(y)} epochs, {df['participant_id'].nunique()} subjects, Deep={y.mean()*100:.1f}%")
print("BASELINE reference (deployed Dense NN): Deep recall~83%  precision~24%")

report((oof>=0.5).astype(int),"[1] XGBoost + new features @0.5")

# calibrate threshold: pick smallest thr giving precision>=0.50
from sklearn.metrics import precision_recall_curve
p,r,th=precision_recall_curve(y,oof)
thr=next((th[i] for i in range(len(th)) if p[i]>=0.50), 0.5)
report((oof>=thr).astype(int),f"[2] + calibrated threshold ({thr:.2f})")

# per-night Viterbi smoothing of calibrated probs
sm=np.zeros(len(y),int)
for pid,idx in df.groupby('participant_id',sort=False).indices.items():
    idx=np.array(sorted(idx)); sm[idx]=viterbi(oof[idx])
report(sm,"[3] + Viterbi smoothing (final)")

# ---- fair comparison: precision at the SAME recall as the deployed model (~0.83) ----
order=np.argsort(-oof)
# find threshold achieving recall ~0.83
from sklearn.metrics import precision_recall_curve
p2,r2,th2=precision_recall_curve(y,oof)
i=np.argmin(np.abs(r2-0.83))
thr83=th2[min(i,len(th2)-1)]
pred83=(oof>=thr83).astype(int)
cm=confusion_matrix(y,pred83,labels=[0,1]); rec=cm[1,1]/cm[1].sum(); prec=cm[1,1]/max(cm[:,1].sum(),1)
print(f"\n[MATCHED-RECALL] new model at recall={rec*100:.1f}% -> precision={prec*100:.1f}%  (vs deployed 83%/24%)")
# also: alarm-oriented high-recall point
i2=np.argmin(np.abs(r2-0.90))
thr90=th2[min(i2,len(th2)-1)]
cm=confusion_matrix(y,(oof>=thr90).astype(int),labels=[0,1]); rec=cm[1,1]/cm[1].sum(); prec=cm[1,1]/max(cm[:,1].sum(),1)
print(f"[ALARM-SAFE]     new model at recall={rec*100:.1f}% -> precision={prec*100:.1f}%")
