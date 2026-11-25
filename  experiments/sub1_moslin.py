"""
Submission #1 — MOS lineal por (hora, píxel) con GFS interpolado en t-1, t, t+1  (ridge, lam=5)
CV (3 folds, constantes oficiales): 0.410 ± 0.021 | F3 (16–30 jun): 0.4035
Hipótesis a contrastar con el LB: ¿la CV con constantes oficiales predice el LB? (el test es más seco).

Uso (misma carpeta que fase2_validacion.py):  python sub1_moslin.py [--data RUTA]
Genera submission_01_moslin.csv
"""
import argparse
import numpy as np
import pandas as pd
from fase2_validacion import (find, hour, bilinear_weights, interp, lag_index, fit_mos, pred_mos,
                              VARS, EVAL_H)

ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
a, _ = ap.parse_known_args()
tr, te, st = (np.load(find(f, a.data)) for f in ["train.npz", "test.npz", "static.npz"])
Xtr, Ytr, ttr = tr["X"].astype(np.float32), tr["Y"].astype(np.float32), tr["times"].astype(str)
Xte, tte = te["X"].astype(np.float32), te["times"].astype(str)
W = bilinear_weights(*(st[k].astype(np.float64) for k in ["gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons"]))
Itr, Ite = interp(Xtr, W), interp(Xte, W)


def window(I, t):
    ip, fp = lag_index(t, -1); ina, fn = lag_index(t, +1)
    print(f"   sin t-1 real: {fp:.2%} | sin t+1 real: {fn:.2%}")
    return np.concatenate([I[ip], I, I[ina]], -1)


print("train:"); Ftr = window(Itr, ttr)
print("test:");  Fte = window(Ite, tte)
htr, hte = hour(ttr), hour(tte)

M = fit_mos(Ytr, Ftr, htr, lam=5.0)          # entrenamiento con TODO el train
ev = np.isin(hte, EVAL_H)
P = np.full(Ite.shape[:3] + (4,), np.nan, np.float32)
P[ev] = pred_mos(M, Fte[ev], hte[ev])
P[..., 0] = np.clip(P[..., 0], 0, 100)
# Recorte al rango observado del WRF en train (la regresión lineal extrapola en pocos píxeles)
lo, hi = Ytr.min((0, 1, 2)), Ytr.max((0, 1, 2))
n_out = ((P[ev] < lo) | (P[ev] > hi)).sum((0, 1, 2))
print("valores fuera del rango de train por variable (se recortan):", dict(zip(VARS, n_out.tolist())))
P = np.clip(P, lo, hi)

# ---------------------------------------------------------------- armado y chequeos de formato
sample = pd.read_csv(find("sample_submission.csv", a.data))
pos = {k: i for i, k in enumerate(tte)}
p = sample["id"].str.split("_", expand=True)
ti = p[0].map(pos).values; r = p[1].astype(int).values; c = p[2].astype(int).values
assert not np.isnan(ti.astype(float)).any(), "hay ids con hora que no está en test.npz"
sub = pd.DataFrame({"id": sample["id"]})
for k, v in enumerate(VARS):
    sub[v] = P[ti.astype(int), r, c, k].round(3)

print("\nCHEQUEOS")
print(" filas:", len(sub), "== sample:", len(sub) == len(sample), "| mismo orden de id:", bool((sub.id == sample.id).all()))
print(" NaN/Inf:", int((~np.isfinite(sub[VARS].values)).sum()))
print(sub[VARS].describe().loc[["mean", "min", "max"]].round(3).to_string())
print("\nMedia predicha vs climatología oficial (sample) por variable — la diferencia es el 'shift' que el modelo aplica:")
print((sub[VARS].mean() - sample[VARS].mean()).round(3).to_string())
mon = pd.to_datetime(sub.id.str[:10], format="%Y%m%d%H").dt.month
print("\nRH predicha por mes (WRF train fue: abr 80.8, may 71.6, jun 61.1):")
print(sub.groupby(mon)["RH"].mean().round(2).to_string())
assert sub["RH"].between(0, 100).all() and sub["T"].between(240, 320).all()
sub.to_csv("submission_01_moslin.csv", index=False)
print("\nGuardado: submission_01_moslin.csv")