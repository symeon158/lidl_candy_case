"""
precompute.py — regenerate precomputed.json (SHAP values + cross-validation scores)
===================================================================================
The app loads these artifacts at runtime so it never has to retrain heavy models on a
page view. Run this only when the data or model changes.

    pip install -r requirements-dev.txt
    python precompute.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor
import shap

URL = ("https://raw.githubusercontent.com/fivethirtyeight/data/master/"
       "candy-power-ranking/candy-data.csv")
FLAGS = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
         "crispedricewafer", "hard", "bar", "pluribus"]
NUM = ["sugarpercent", "pricepercent"]
FEATS = FLAGS + NUM
NICE = {"chocolate": "Chocolate", "fruity": "Fruity (gummy)", "caramel": "Caramel",
        "peanutyalmondy": "Peanut / almond", "nougat": "Nougat",
        "crispedricewafer": "Crispy wafer", "hard": "Hard candy", "bar": "Bar format",
        "pluribus": "Multi-pack", "sugarpercent": "Sweetness", "pricepercent": "Price"}

df = pd.read_csv(URL)
df = df[df[FLAGS].sum(axis=1) > 0].reset_index(drop=True)
X, y = df[FEATS], df["winpercent"]

# 1) repeated-CV bake-off
models = {
    "OLS": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 60),
                                                     max_iter=20000, cv=5, random_state=0)),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=0),
}
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
cv = {name: [round(float(v), 4) for v in cross_val_score(m, X, y, cv=rkf, scoring="r2")]
      for name, m in models.items()}

# 2) SHAP on a gradient-boosted model
cols = [NICE[c] for c in FEATS]
Xs = X.copy(); Xs.columns = cols
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                random_state=42).fit(Xs, y)
sv = shap.TreeExplainer(gbm)(Xs)
bee = {}
for j, col in enumerate(cols):
    v = X.iloc[:, j].values.astype(float)
    norm = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)
    bee[col] = {"shap": [round(float(s), 3) for s in sv.values[:, j]],
                "fval": [round(float(f), 3) for f in norm]}
mean_abs = {col: round(float(np.abs(sv.values[:, j]).mean()), 3) for j, col in enumerate(cols)}

rec = {c: 0 for c in cols}
rec.update({"Chocolate": 1, "Peanut / almond": 1, "Crispy wafer": 1,
            "Bar format": 1, "Sweetness": 0.70, "Price": 0.45})
svr = shap.TreeExplainer(gbm)(pd.DataFrame([rec])[cols])
wf = sorted([{"feature": cols[i], "shap": round(float(svr.values[0, i]), 3)} for i in range(len(cols))],
            key=lambda d: -abs(d["shap"]))

art = {"cv": cv, "shap_beeswarm": bee, "shap_mean_abs": mean_abs,
       "shap_waterfall": {"base": round(float(svr.base_values[0]), 2),
                          "contribs": wf,
                          "final": round(float(svr.base_values[0] + svr.values[0].sum()), 1)}}
json.dump(art, open("precomputed.json", "w"), separators=(",", ":"))
print("wrote precomputed.json")
