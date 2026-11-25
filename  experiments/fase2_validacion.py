"""
Hackathon IA PUCP 2026 · FASE 2 — Marco de validación + hipótesis + baselines sin red neuronal
==============================================================================================
Corre en CPU en ~1–3 min. No genera submission (eso viene después de leer los resultados).

  A. ¿sample_submission = climatología por (hora, píxel) de TODO el train?   -> qué es exactamente "score 0"
  B. H1: ¿el WRF se reinicia cada día a 00 UTC?  (saltos 23->00 UTC en Y)    -> features "hora desde init"
  C. CV temporal: 3 bloques de 15 días (mayo 17–31, jun 1–15, jun 16–30), ventana expansiva,
     embargo de 1 día, evaluando solo horas 00,03,…,21. Score con constantes OFICIALES y local.
  D. Baselines: climatología · GFS+sesgo(píxel,hora) · MOS lineal por píxel-hora con GFS(t)
     · MOS lineal con GFS(t-1,t,t+1)  -> H2: ¿el GFS "futuro" t+1 ayuda?

Uso:  python fase2_validacion.py [--data RUTA]
"""
import argparse, glob, sys
import numpy as np
import pandas as pd

VARS = ["RH", "T", "U", "V"]
CONST = np.array([25.360, 2.0652, 2.6776, 2.0876])
EVAL_H = [0, 3, 6, 9, 12, 15, 18, 21]
FOLDS = [("F1 may17-31", "2024051700", "2024053123"),
         ("F2 jun01-15", "2024060100", "2024061523"),
         ("F3 jun16-30", "2024061600", "2024063023")]
EMBARGO_H = 24


def find(name, root=None):
    for p in ([f"{root}/**/{name}"] if root else []) + [f"/kaggle/input/**/{name}", f"../data/**/{name}",
                                                        f"data/**/{name}", f"**/{name}"]:
        h = glob.glob(p, recursive=True)
        if h:
            return h[0]
    sys.exit(f"No encuentro {name}")


def hdr(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)
def hour(t): return np.array([int(k[-2:]) for k in t])
def rmse(p, y): return np.sqrt(((p - y) ** 2).mean(axis=(0, 1, 2)))


def bilinear_weights(glat, glon, wlat, wlon):
    asc = glat[0] > glat[-1]; la = glat[::-1] if asc else glat
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    fi = np.interp(LAT, la, np.arange(len(la))); fj = np.interp(LON, glon, np.arange(len(glon)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(la) - 2); di = fi - i0
    j0 = np.clip(np.floor(fj).astype(int), 0, len(glon) - 2); dj = fj - j0
    I0, I1 = ((len(la) - 1 - i0, len(la) - 2 - i0) if asc else (i0, i0 + 1))
    return I0, I1, j0, j0 + 1, di, dj


def interp(X, W):
    I0, I1, J0, J1, di, dj = W; di, dj = di[None, ..., None], dj[None, ..., None]
    return ((1 - di) * (1 - dj) * X[:, I0, J0] + di * (1 - dj) * X[:, I1, J0]
            + (1 - di) * dj * X[:, I0, J1] + di * dj * X[:, I1, J1]).astype(np.float32)


def lag_index(t, lag):
    """Índice de la hora t+lag; si no existe (hueco), usa la propia hora t."""
    ts = pd.to_datetime(t, format="%Y%m%d%H"); pos = {k: i for i, k in enumerate(ts)}
    idx = np.array([pos.get(k + pd.Timedelta(hours=lag), i) for i, k in enumerate(ts)])
    return idx, float(np.mean(idx == np.arange(len(t))))   # fracción SIN vecino real (rellenada con t)


# ----------------------------------------------------------------------------- modelos baseline
def fit_clim(Y, h):
    return {k: Y[h == k].mean(0) for k in EVAL_H}


def fit_bias(Y, I, h):
    return {k: (Y[h == k] - I[h == k]).mean(0) for k in EVAL_H}


def fit_mos(Y, F, h, lam=1.0):
    """Regresión ridge independiente por (hora, píxel, variable objetivo).
    F: (n, 70, 35, p) features estandarizadas. Devuelve dict hora -> (beta (70,35,p+1,4), mu, sd)."""
    out = {}
    for k in EVAL_H:
        m = h == k; Fk, Yk = F[m], Y[m]
        mu, sd = Fk.mean(0), Fk.std(0) + 1e-6
        Z = np.concatenate([np.ones(Fk.shape[:-1] + (1,), np.float32), (Fk - mu) / sd], -1)   # (n,70,35,q)
        A = np.einsum("nijp,nijq->ijpq", Z, Z)
        A += lam * np.eye(Z.shape[-1])[None, None] * np.r_[0, np.ones(Z.shape[-1] - 1)][None, None, :, None]
        b = np.einsum("nijp,nijv->ijpv", Z, Yk)
        out[k] = (np.linalg.solve(A, b), mu, sd)
    return out


def pred_mos(M, F, h):
    P = np.empty(F.shape[:3] + (4,), np.float32)
    for k in EVAL_H:
        m = h == k
        if not m.any():
            continue
        beta, mu, sd = M[k]
        Z = np.concatenate([np.ones(F[m].shape[:-1] + (1,), np.float32), (F[m] - mu) / sd], -1)
        P[m] = np.einsum("nijq,ijqv->nijv", Z, beta)
    return P


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
    a, _ = ap.parse_known_args()
    tr, te, st = (np.load(find(f, a.data)) for f in ["train.npz", "test.npz", "static.npz"])
    Xtr, Ytr, ttr = tr["X"].astype(np.float32), tr["Y"].astype(np.float32), tr["times"].astype(str)
    tte = te["times"].astype(str)
    glat, glon, wlat, wlon = (st[k].astype(np.float64) for k in ["gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons"])
    h = hour(ttr)
    Itr = interp(Xtr, bilinear_weights(glat, glon, wlat, wlon))

    # ------------------------------------------------------------------ A
    hdr("A. ¿Qué es exactamente la 'climatología' del sample_submission?")
    sample = pd.read_csv(find("sample_submission.csv", a.data))
    p = sample["id"].str.split("_", expand=True)
    sh, sr, sc = p[0].str[-2:].astype(int).values, p[1].astype(int).values, p[2].astype(int).values
    S = sample[VARS].values
    cands = {
        "media por (hora,píxel) de TODO el train": np.stack([Ytr[h == k].mean(0) for k in range(24)]),
        "media por píxel (sin hora) de TODO el train": np.repeat(Ytr.mean(0)[None], 24, 0),
        "media por (hora,píxel) train hasta 15-jun": np.stack([Ytr[(h == k) & (ttr < "2024061600")].mean(0) for k in range(24)]),
    }
    for nm, C in cands.items():
        d = np.abs(C[sh, sr, sc] - S)
        print(f"{nm:45s} max|dif| = {d.max(0).round(4)}   media|dif| = {d.mean(0).round(4)}")
    print("¿El sample varía entre días con la misma hora? ",
          bool(sample.groupby([p[0].str[-2:], p[1], p[2]])[VARS].nunique().max().max() > 1))

    # ------------------------------------------------------------------ B
    hdr("B. H1 — ¿reinicio diario del WRF a 00 UTC?  |Δ1h| medio de dominio por hora de llegada")
    ts = pd.to_datetime(ttr, format="%Y%m%d%H")
    cons = np.where(np.diff(ts.values).astype("timedelta64[h]").astype(int) == 1)[0] + 1   # i con i-1 = 1h antes
    dY = np.abs(Ytr[cons] - Ytr[cons - 1]).mean((1, 2))      # (m,4)
    dX = np.abs(Xtr[cons] - Xtr[cons - 1]).mean((1, 2))
    hc = h[cons]
    r_int = np.stack([np.sqrt(((Itr[h == k] - Ytr[h == k]) ** 2).mean((0, 1, 2))) for k in range(24)])
    rows = []
    for k in range(24):
        m = hc == k
        rows.append([k, (k - 5) % 24] + list(dY[m].mean(0)) + list(dX[m].mean(0)[[1, 2]]) + list(r_int[k][[1, 2]]))
    df = pd.DataFrame(rows, columns=["UTC", "Lima", "dY_RH", "dY_T", "dY_U", "dY_V", "dX_T", "dX_U",
                                     "rmseInterp_T", "rmseInterp_U"])
    print(df.round(3).to_string(index=False))
    med = df[["dY_RH", "dY_T", "dY_U", "dY_V"]].median()
    print("salto en 00 UTC / mediana de las otras horas:",
          (df.loc[df.UTC == 0, ["dY_RH", "dY_T", "dY_U", "dY_V"]].iloc[0] / med).round(2).to_dict())
    print("-> si el ratio de 00 UTC es >>1 en Y pero no en X, el WRF arranca cada día y 00 UTC ≈ condición inicial.")
    # huecos: ¿son días completos 00–23 UTC?
    full_days_missing = sorted(set(pd.date_range(ts[0].normalize(), ts[-1].normalize(), freq="D")) -
                               set(ts.normalize()))
    print(f"días sin NINGUNA hora en train: {[str(d.date()) for d in full_days_missing]}")
    cnt = pd.Series(ts.normalize()).value_counts()
    print(f"días con datos parciales (<24 h): {len(cnt[cnt < 24])}  -> {sorted([(str(d.date()), int(c)) for d, c in cnt[cnt < 24].items()])}")

    # ------------------------------------------------------------------ C + D
    hdr("C/D. CV TEMPORAL (3 bloques, ventana expansiva, embargo 24 h) — baselines")
    ip, fp = lag_index(ttr, -1); ip1, fn = lag_index(ttr, +1)
    print(f"fracción de horas sin t-1 real: {fp:.3%} | sin t+1 real: {fn:.3%} (se rellenan con t)")
    F_t = Itr                                                            # (n,70,35,4)
    F_win = np.concatenate([Itr[ip], Itr, Itr[ip1]], -1)                 # (n,70,35,12)

    results = []; per_hour = {}
    for nm, v0, v1 in FOLDS:
        val = (ttr >= v0) & (ttr <= v1)
        emb = (pd.to_datetime(v0, format="%Y%m%d%H") - pd.Timedelta(hours=EMBARGO_H)).strftime("%Y%m%d%H")
        trn = ttr < emb
        ev = val & np.isin(h, EVAL_H)
        clim = fit_clim(Ytr[trn], h[trn]); P_clim = np.stack([clim[k] for k in h[ev]])
        r_clim = rmse(P_clim, Ytr[ev])
        preds = {"clim": P_clim}
        bias = fit_bias(Ytr[trn], Itr[trn], h[trn])
        preds["GFS+sesgo"] = Itr[ev] + np.stack([bias[k] for k in h[ev]])
        preds["MOS-lin GFS(t)"] = pred_mos(fit_mos(Ytr[trn], F_t[trn], h[trn]), F_t[ev], h[ev])
        preds["MOS-lin GFS(t-1,t,t+1)"] = pred_mos(fit_mos(Ytr[trn], F_win[trn], h[trn], lam=5.0), F_win[ev], h[ev])
        for mn, P in preds.items():
            P = P.copy(); P[..., 0] = np.clip(P[..., 0], 0, 100)
            r = rmse(P, Ytr[ev])
            results.append([nm, int(trn.sum()), int(ev.sum()), mn, *r,
                            1 - (r / CONST).mean(), 1 - (r / r_clim).mean()])
            per_hour[(nm, mn)] = np.stack([rmse(P[h[ev] == k], Ytr[ev][h[ev] == k]) for k in EVAL_H])
    R = pd.DataFrame(results, columns=["fold", "n_train", "n_eval", "modelo", *[f"rmse_{v}" for v in VARS],
                                       "score_oficial", "score_local"])
    print(R.round(4).to_string(index=False))
    hdr("Promedio sobre folds (y F3 sola, la más parecida a julio)")
    agg = R.groupby("modelo")[[f"rmse_{v}" for v in VARS] + ["score_oficial", "score_local"]].mean()
    agg["score_oficial_F3"] = R[R.fold.str.startswith("F3")].set_index("modelo")["score_oficial"]
    agg["std_score_oficial"] = R.groupby("modelo")["score_oficial"].std()
    print(agg.sort_values("score_oficial").round(4).to_string())

    hdr("Pérdida por variable (rmse/const) del mejor modelo — de dónde salen los puntos")
    best = agg["score_oficial"].idxmax()
    loss = (agg.loc[best, [f"rmse_{v}" for v in VARS]].values / CONST)
    print(f"{best}: " + "  ".join(f"{v}={x:.3f}" for v, x in zip(VARS, loss)) + f"  | suma/4 = {loss.mean():.3f}")

    hdr(f"RMSE por hora UTC (promedio de folds) — {best}")
    ph = np.mean([per_hour[(f[0], best)] for f in FOLDS], 0)
    print(pd.DataFrame(ph, index=[f"{k:02d} UTC ({(k - 5) % 24:02d} Lima)" for k in EVAL_H], columns=VARS).round(3).to_string())


if __name__ == "__main__":
    main()