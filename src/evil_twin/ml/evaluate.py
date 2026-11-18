"""Evaluation with honest undefined-metric handling."""
from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import (confusion_matrix,precision_score,recall_score,f1_score,balanced_accuracy_score,
                             average_precision_score,roc_auc_score)


def evaluate_predictions(y, probabilities, threshold=.5, abstention_margin=0.0):
    y=np.asarray(y); p=np.asarray(probabilities)
    decided=np.abs(p-threshold)>=abstention_margin
    pred=(p[decided]>=threshold).astype(int); truth=y[decided]
    out={"n":len(y),"decisions":int(decided.sum()),"decision_coverage":float(decided.mean()) if len(y) else 0,
         "abstention_rate":float(1-decided.mean()) if len(y) else 1}
    if len(truth)==0 or len(set(truth))<2:
        out.update({k:None for k in ["precision","recall","f1","balanced_accuracy","pr_auc","roc_auc","legitimate_fpr","verified_attack_recall"]})
        out["confusion_matrix"]=None; return out
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    out.update(confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],precision=precision_score(truth,pred,zero_division=0),
               recall=recall_score(truth,pred),f1=f1_score(truth,pred),balanced_accuracy=balanced_accuracy_score(truth,pred),
               pr_auc=average_precision_score(truth,p[decided]),roc_auc=roc_auc_score(truth,p[decided]),
               legitimate_fpr=fp/(fp+tn),verified_attack_recall=tp/(tp+fn))
    return out


def timed_probabilities(model, X):
    start=time.perf_counter(); p=model.predict_proba(X)[:,1]; elapsed=time.perf_counter()-start
    return p, elapsed, elapsed/max(len(X),1)
