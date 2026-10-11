"""Predicting eruptive events at Kilauea from earthquake data.
Tasks: (1) classify erupting vs repose, (2) regress time-to-eruption.
Models: logistic regression, k-means, random forest, neural network (PyTorch)."""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (cohen_kappa_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score)
warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--eq", default="earthquakes.csv")
p.add_argument("--er", default="eruptions.csv")
p.add_argument("--out", default="results")
p.add_argument("--split", choices=["random", "time"], default="random",
               help="random = like the paper; time = train on past, test on future (more honest)")
p.add_argument("--start-year", type=int, default=None, help="keep quakes from this year onward")
p.add_argument("--end-year", type=int, default=None, help="keep quakes up to and including this year")
p.add_argument("--nn-epochs", type=int, default=30)
p.add_argument("--no-nn", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
SEED = 0
np.random.seed(SEED)

# ---------- 1. Load + filter ----------
eq = pd.read_csv(args.eq, parse_dates=["time"])
eq = eq[eq.latitude.between(19.1, 19.5) & eq.longitude.between(-155.5, -155.0)]
eq = eq.sort_values("time").reset_index(drop=True)
er = pd.read_csv(args.er, parse_dates=["start", "end"]).sort_values("start").reset_index(drop=True)
print(f"[1] {len(eq)} earthquakes, {len(er)} eruptions")

# ---------- 2. Labels ----------
t = eq.time.values.astype("datetime64[ns]")
st, en = er.start.values.astype("datetime64[ns]"), er.end.values.astype("datetime64[ns]")
i = np.searchsorted(st, t, side="right") - 1            # last eruption that started before t
erupting = (i >= 0) & (t <= en[np.clip(i, 0, None)])
j = np.searchsorted(st, t, side="right")                # next eruption start after t
has_next = j < len(st)
tte = np.where(erupting, 0.0,
               (st[np.clip(j, 0, len(st) - 1)] - t) / np.timedelta64(1, "D"))
eq["erupting"] = erupting.astype(int)
eq["tte"] = tte
eq = eq[has_next | erupting].reset_index(drop=True)      # drop quakes after the last eruption
print(f"[2] {eq.erupting.mean()*100:.1f}% of quakes occur during eruptions")

# ---------- 3. Rate features (backward-looking only) ----------
tt = eq.time.values.astype("datetime64[ns]")
for name, d in [("rate_1d", 1), ("rate_7d", 7), ("rate_30d", 30)]:
    lo = np.searchsorted(tt, tt - np.timedelta64(d, "D"), side="left")
    eq[name] = np.arange(len(tt)) - lo                   # quakes in [t-d, t)
features = ["longitude", "latitude", "depth", "mag", "rate_1d", "rate_7d", "rate_30d"]
print("[3] features:", features)

# ---------- 3b. Optional year filter (labels/rates above use the FULL history, so edges stay correct) ----------
if args.start_year is not None:
    eq = eq[eq.time.dt.year >= args.start_year]
if args.end_year is not None:
    eq = eq[eq.time.dt.year <= args.end_year]
eq = eq.reset_index(drop=True)
if args.start_year is not None or args.end_year is not None:
    print(f"[3b] year filter {args.start_year}-{args.end_year}: {len(eq)} quakes kept, "
          f"{eq.erupting.sum()} during eruptions")
if len(eq) < 50 or eq.erupting.sum() < 10 or (eq.erupting == 0).sum() < 10:
    raise SystemExit("Too few quakes (or only one class: all erupting / all repose) in this year range. "
                     "Choose a wider range that includes both eruption and repose periods.")

# ---------- 4. Split + scale ----------
X, yc, yr = eq[features].values, eq.erupting.values, eq.tte.values
if args.split == "random":
    X_tr, X_tmp, yc_tr, yc_tmp, yr_tr, yr_tmp = train_test_split(
        X, yc, yr, test_size=0.30, stratify=yc, random_state=SEED)
    X_dev, X_te, yc_dev, yc_te, yr_dev, yr_te = train_test_split(
        X_tmp, yc_tmp, yr_tmp, test_size=1/3, stratify=yc_tmp, random_state=SEED)
else:
    a, b = int(.7 * len(X)), int(.9 * len(X))
    X_tr, X_dev, X_te = X[:a], X[a:b], X[b:]
    yc_tr, yc_dev, yc_te = yc[:a], yc[a:b], yc[b:]
    yr_tr, yr_dev, yr_te = yr[:a], yr[a:b], yr[b:]
sx = StandardScaler().fit(X_tr)
X_tr, X_dev, X_te = [sx.transform(v) for v in (X_tr, X_dev, X_te)]
sy = StandardScaler().fit(yr_tr.reshape(-1, 1))
yr_tr, yr_dev, yr_te = [sy.transform(v.reshape(-1, 1)).ravel() for v in (yr_tr, yr_dev, yr_te)]
print(f"[4] split={args.split}: train={len(X_tr)} dev={len(X_dev)} test={len(X_te)}")

# ---------- helpers ----------
rows = []
def evaluate(model, split, y, pred, score):
    try: auc = roc_auc_score(y, score)
    except ValueError: auc = float("nan")
    k = cohen_kappa_score(y, pred)
    cm = confusion_matrix(y, pred).tolist()
    rows.append(dict(model=model, split=split, kappa=round(k, 3), auroc=round(auc, 3), confusion=cm))
    return k, auc

def report_cls(model, preds):
    for split, y, (pred, score) in zip(["train", "dev", "test"], [yc_tr, yc_dev, yc_te], preds):
        evaluate(model, split, y, pred, score)
    r = rows[-1]; print(f"    {model:20s} test kappa={r['kappa']} AUROC={r['auroc']}")

# ---------- 5. Classification ----------
print("[5] Classification")
Xs = [X_tr, X_dev, X_te]

lr = LogisticRegression(max_iter=1000).fit(X_tr, yc_tr)
report_cls("LogisticRegression", [(lr.predict(x), lr.predict_proba(x)[:, 1]) for x in Xs])

# K-means: elbow plot on repose quakes, then 8 clusters per class (paper's choice)
ks = range(2, 16)
inertia = [KMeans(k, n_init=5, random_state=SEED).fit(X_tr[yc_tr == 0]).inertia_ for k in ks]
plt.figure(); plt.plot(list(ks), inertia, "bx-"); plt.xlabel("clusters"); plt.ylabel("inertia")
plt.title("Repose periods"); plt.savefig(f"{args.out}/elbow_kmeans.png", dpi=150); plt.close()
km0 = KMeans(8, n_init=10, random_state=SEED).fit(X_tr[yc_tr == 0])
km1 = KMeans(8, n_init=10, random_state=SEED).fit(X_tr[yc_tr == 1])
def km_predict(x):
    d0, d1 = km0.transform(x).min(1), km1.transform(x).min(1)
    return (d1 < d0).astype(int), d0 - d1
report_cls("KMeans", [km_predict(x) for x in Xs])

# Random forest: tune depth on dev AUROC
depths, tr_auc, dev_auc = list(range(2, 31, 2)), [], []
for d in depths:
    m = RandomForestClassifier(200, max_depth=d, max_features="sqrt", random_state=SEED, n_jobs=-1).fit(X_tr, yc_tr)
    tr_auc.append(roc_auc_score(yc_tr, m.predict_proba(X_tr)[:, 1]))
    dev_auc.append(roc_auc_score(yc_dev, m.predict_proba(X_dev)[:, 1]))
best_d = depths[int(np.argmax(dev_auc))]
plt.figure(); plt.plot(depths, tr_auc, "b", label="Train AUC"); plt.plot(depths, dev_auc, "r", label="Dev AUC")
plt.xlabel("tree depth"); plt.ylabel("AUROC"); plt.legend(); plt.title("Tree depth tuning")
plt.savefig(f"{args.out}/rf_depth_tuning.png", dpi=150); plt.close()
rf = RandomForestClassifier(200, max_depth=best_d, max_features="sqrt", random_state=SEED, n_jobs=-1).fit(X_tr, yc_tr)
print(f"    best RF depth = {best_d}")
report_cls("RandomForest", [(rf.predict(x), rf.predict_proba(x)[:, 1]) for x in Xs])

# ---------- Neural network (optional) ----------
nn_ok = False
if not args.no_nn:
    try:
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(SEED); nn_ok = True
    except ImportError:
        print("    torch not installed -> skipping neural network (pip install torch)")

def make_net(d_in):
    layers, d = [], d_in
    for _ in range(4):
        layers += [nn.Linear(d, 1000), nn.ReLU()]; d = 1000
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def train_net(task, Xtr, ytr, Xdv, ydv, epochs, lr=1e-3, wd=1e-4):
    net = make_net(Xtr.shape[1])
    if task == "cls":   # weighted loss handles the ~80/20 imbalance
        pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.float32)), batch_size=50, shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(net(xb).squeeze(1), yb).backward(); opt.step()
    net.eval(); return net

def net_out(net, x):
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()

if nn_ok:
    net = train_net("cls", X_tr, yc_tr, X_dev, yc_dev, args.nn_epochs)
    def nn_cls(x):
        s = 1 / (1 + np.exp(-net_out(net, x))); return (s > 0.5).astype(int), s
    report_cls("NeuralNet", [nn_cls(x) for x in Xs])

# ---------- 6. Regression ----------
print("[6] Time-to-eruption regression")
reg_rows = []
def report_reg(model, preds):
    out = {"model": model}
    for split, y, pr in zip(["train", "dev", "test"], [yr_tr, yr_dev, yr_te], preds):
        out[f"rmse_{split}"] = round(mean_squared_error(y, pr) ** 0.5, 3)
        out[f"r2_{split}"] = round(r2_score(y, pr), 3)
    reg_rows.append(out); print(f"    {model:14s} test RMSE={out['rmse_test']} R2={out['r2_test']}")

# K-means: cluster on [features, time]; predict centroid time of nearest cluster (features only)
km = KMeans(10, n_init=10, random_state=SEED).fit(np.c_[X_tr, yr_tr])
C = km.cluster_centers_
def km_reg(x): return C[((x[:, None, :] - C[None, :, :-1]) ** 2).sum(-1).argmin(1), -1]
report_reg("KMeans", [km_reg(x) for x in Xs])

# Random forest regressor: tune depth on dev R2
r2s = []
for d in depths:
    m = RandomForestRegressor(200, max_depth=d, random_state=SEED, n_jobs=-1).fit(X_tr, yr_tr)
    r2s.append(r2_score(yr_dev, m.predict(X_dev)))
rfr = RandomForestRegressor(200, max_depth=depths[int(np.argmax(r2s))], random_state=SEED, n_jobs=-1).fit(X_tr, yr_tr)
report_reg("RandomForest", [rfr.predict(x) for x in Xs])

if nn_ok:
    netr = train_net("reg", X_tr, yr_tr, X_dev, yr_dev, args.nn_epochs)
    report_reg("NeuralNet", [net_out(netr, x) for x in Xs])

# ---------- 7. Save outputs ----------
imp = rfr.feature_importances_; o = np.argsort(imp)[::-1]
plt.figure(figsize=(8, 4)); plt.bar(np.array(features)[o], imp[o], color="r")
plt.xticks(rotation=30); plt.title("Random forest feature importance")
plt.tight_layout(); plt.savefig(f"{args.out}/feature_importance.png", dpi=150); plt.close()

pr = rfr.predict(X_te)
lim = [min(pr.min(), yr_te.min()), max(pr.max(), yr_te.max())]
plt.figure(); plt.scatter(pr, yr_te, s=4, c="r"); plt.plot(lim, lim, "b--")
plt.xlabel("Predicted time-to-eruption"); plt.ylabel("Observed"); plt.title("Test set (RF)")
plt.savefig(f"{args.out}/pred_vs_observed.png", dpi=150); plt.close()

pd.DataFrame(rows).to_csv(f"{args.out}/classification_results.csv", index=False)
pd.DataFrame(reg_rows).to_csv(f"{args.out}/regression_results.csv", index=False)
print(f"[7] Saved tables and plots in ./{args.out}/")
