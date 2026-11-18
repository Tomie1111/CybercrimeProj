"""Fresh model fitting with validation-only model and threshold selection."""
from __future__ import annotations
import itertools, json, hashlib
from pathlib import Path
import joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .preprocessing import infer_schema,build_preprocessor,validate_schema
from .evaluate import evaluate_predictions


def candidates(seed=20261116):
    for depth,leaf in itertools.product([3,6,None],[1,5]):
        yield "cart",{"max_depth":depth,"min_samples_leaf":leaf},DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight="balanced",random_state=seed)
    for n,depth,leaf in itertools.product([100,300],[6,None],[1,5]):
        yield "random_forest",{"n_estimators":n,"max_depth":depth,"min_samples_leaf":leaf},RandomForestClassifier(n_estimators=n,max_depth=depth,min_samples_leaf=leaf,class_weight="balanced",n_jobs=-1,random_state=seed)
    for c in [.1,1.,10.]: yield "logistic_regression",{"C":c},LogisticRegression(C=c,class_weight="balanced",max_iter=2000,random_state=seed)


def train_select(train, validation, model_dir, label="label", seed=20261116, objective="balanced_accuracy"):
    if set(train[label].dropna().unique()) != {0,1}: raise ValueError("Training requires both independently verified classes")
    schema=infer_schema(train.drop(columns=[label])); Xtr=train.drop(columns=[label]); ytr=train[label].astype(int)
    Xv=validation.drop(columns=[label]); yv=validation[label].astype(int); validate_schema(Xv,schema)
    rows=[]; best=None
    for name,params,clf in candidates(seed):
        pipe=Pipeline([("preprocess",build_preprocessor(schema,scale=name=="logistic_regression")),("classifier",clf)])
        pipe.fit(Xtr,ytr); probs=pipe.predict_proba(Xv)[:,1]
        for threshold in [.3,.4,.5,.6,.7]:
            metrics=evaluate_predictions(yv,probs,threshold); row={"model":name,"params":json.dumps(params),"threshold":threshold,**metrics}; rows.append(row)
            score=metrics.get(objective)
            key=(-1 if score is None else score, -metrics.get("legitimate_fpr",1), metrics["decision_coverage"])
            if best is None or key>best[0]: best=(key,pipe,name,params,threshold,metrics)
    model_dir=Path(model_dir); model_dir.mkdir(parents=True,exist_ok=True)
    artifact={"pipeline":best[1],"schema":schema.as_dict(),"threshold":best[4],"seed":seed,"selection_split":"validation","objective":objective,
              "model":best[2],"params":best[3]}
    path=model_dir/"selected_model.joblib"; joblib.dump(artifact,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); (model_dir/"selected_model.metadata.json").write_text(json.dumps({k:v for k,v in artifact.items() if k!="pipeline"}|{"sha256":digest},indent=2))
    return pd.DataFrame(rows),artifact
