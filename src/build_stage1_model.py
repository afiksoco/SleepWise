import os,sys,json,numpy as np,pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.insert(0,os.path.dirname(__file__))
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

BASE=['hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
 'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
 'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
 'hr_stability','sleep_cycle_position','acc_std','acc_move_ratio']

def add_causal(d):
    out=[]
    for pid,sub in d.groupby('participant_id',sort=False):
        sub=sub.copy().reset_index(drop=True)
        sub['min_since_onset']=np.arange(len(sub))
        em=sub['hr_mean'].expanding().mean(); es=sub['hr_mean'].expanding().std().fillna(1)+1e-6
        sub['hr_z']=(sub['hr_mean']-em)/es                       # CAUSAL personalized baseline
        sub['hr_drop_from_max']=sub['hr_mean'].cummax()-sub['hr_mean']
        for w in (5,10,15):
            sub[f'hr_rmean{w}']=sub['hr_mean'].rolling(w,min_periods=1).mean()
            sub[f'hr_rstd{w}'] =sub['hr_mean'].rolling(w,min_periods=1).std().fillna(0)
            sub[f'acc_rmean{w}']=sub['acc_std'].rolling(w,min_periods=1).mean()
        out.append(sub)
    return pd.concat(out,ignore_index=True)

NEW=['min_since_onset','hr_z','hr_drop_from_max']+[f'hr_rmean{w}' for w in(5,10,15)]+\
    [f'hr_rstd{w}' for w in(5,10,15)]+[f'acc_rmean{w}' for w in(5,10,15)]
FEATS=BASE+NEW

df=add_causal(pd.read_pickle('sleepaccel_feats.pkl'))
y=(simplify_labels(df['label_raw'].values,'binary_n3')=='Deep').astype(int)
g=df['participant_id'].values
X=df[FEATS].fillna(0).values.astype('float32')
print(f"{len(y)} epochs, {df['participant_id'].nunique()} subj, {len(FEATS)} features, Deep={y.mean()*100:.1f}%")

def make(n):
    m=tf.keras.Sequential([tf.keras.layers.Input((n,)),
        tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2,activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy')
    return m

oof=np.zeros(len(y))
for tr,te in GroupKFold(5).split(X,y,g):
    sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
    cw=compute_class_weight('balanced',classes=np.array([0,1]),y=y[tr]); cw={0:cw[0],1:cw[1]*0.6}
    m=make(len(FEATS))
    m.fit(Xtr,y[tr],validation_split=0.15,epochs=60,batch_size=64,class_weight=cw,verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)])
    oof[te]=m.predict(Xte,verbose=0)[:,1]

def at_recall(target):
    p,r,th=precision_recall_curve(y,oof); i=int(np.argmin(np.abs(r-target)))
    thr=th[min(i,len(th)-1)]; pred=(oof>=thr).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    return cm[1,1]/cm[1].sum(), cm[1,1]/max(cm[:,1].sum(),1), thr
for t in (0.83,0.60):
    rec,prec,thr=at_recall(t); print(f"  Dense-NN(new feats) @recall~{t}: recall={rec*100:.1f}% precision={prec*100:.1f}% thr={thr:.2f}")

# save OOF for the figure
np.savez('models/stage1_oof.npz',y=y,oof=oof)

# ---- ship model on ALL data ----
sc=StandardScaler(); Xs=sc.fit_transform(X)
cw=compute_class_weight('balanced',classes=np.array([0,1]),y=y); cw={0:cw[0],1:cw[1]*0.6}
model=make(len(FEATS))
model.fit(Xs,y,validation_split=0.1,epochs=80,batch_size=64,class_weight=cw,verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])
OUT='models/walch_binary_v2'; os.makedirs(OUT,exist_ok=True)
model.export(OUT+'/saved')
tflite=tf.lite.TFLiteConverter.from_saved_model(OUT+'/saved').convert()
open(OUT+'/sleep_stage_model.tflite','wb').write(tflite)
# calibrated threshold for recall~0.83 operating point (alarm-safe, matched to old)
_,_,thr83=at_recall(0.83)
json.dump({'feature_names':FEATS,'class_names':['Light','Deep'],
  'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist(),
  'input_shape':[1,len(FEATS)],'output_shape':[1,2],'deep_index':1,
  'operating_threshold':float(thr83),'label_scheme':'binary_n3',
  'trained_on':'Walch 2019; causal features; temp removed'},
  open(OUT+'/tflite_metadata.json','w'),indent=2)
print(f"EXPORTED {OUT}/ ({len(FEATS)} features, thr={thr83:.2f})")
