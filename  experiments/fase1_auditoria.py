"""
Hackathon IA PUCP 2026 · FASE 1 — Auditoría de datos + EDA físico
==================================================================
No entrena nada. Solo mide. Imprime un reporte de texto (pégamelo completo)
y guarda figuras en ./fase1_figs/.

Uso:
    python fase1_auditoria.py                      # busca la data sola (Kaggle, ../data, data/)
    python fase1_auditoria.py --data /ruta/a/data
En Kaggle/Colab: copia el archivo a una celda y córrela (los argumentos son opcionales).
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

VARS = ["RH", "T", "U", "V"]
UNITS = ["%", "K", "m/s", "m/s"]
CONST = np.array([25.360, 2.0652, 2.6776, 2.0876])   # constantes oficiales del leaderboard
NPIX = 70 * 35


# ----------------------------------------------------------------------------- utilidades
def find(name, root=None):
    pats = ([f"{root}/**/{name}"] if root else []) + [
        f"/kaggle/input/**/{name}", f"../data/**/{name}", f"data/**/{name}", f"**/{name}"]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    sys.exit(f"No encuentro {name}")


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def hour(t):
    return np.array([int(k[-2:]) for k in t])


def rmse(p, y):
    return np.sqrt(((p - y) ** 2).mean(axis=(0, 1, 2)))


def score(r, norm):
    return 1 - (r / norm).mean()


def fmt(r):
    return "  ".join(f"{v}={x:.3f}" for v, x in zip(VARS, r))


def bilinear_weights(glat, glon, wlat, wlon):
    """Pesos bilineales GFS(17x13, fila0=norte) -> WRF(70x35, fila0=sur), por coordenadas."""
    # pasamos GFS a latitud ascendente para buscar índices; luego remapeamos a la fila original
    asc = glat[0] > glat[-1]
    la = glat[::-1] if asc else glat
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    fi = np.interp(LAT, la, np.arange(len(la)))
    fj = np.interp(LON, glon, np.arange(len(glon)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(la) - 2); di = fi - i0
    j0 = np.clip(np.floor(fj).astype(int), 0, len(glon) - 2); dj = fj - j0
    if asc:  # índice en el arreglo original (fila 0 = norte)
        I0, I1 = len(la) - 1 - i0, len(la) - 2 - i0
    else:
        I0, I1 = i0, i0 + 1
    return I0, I1, j0, j0 + 1, di, dj


def interp(X, W):
    I0, I1, J0, J1, di, dj = W
    di, dj = di[None, ..., None], dj[None, ..., None]
    return ((1 - di) * (1 - dj) * X[:, I0, J0] + di * (1 - dj) * X[:, I1, J0]
            + (1 - di) * dj * X[:, I0, J1] + di * dj * X[:, I1, J1]).astype(np.float32)


def smooth(a, k):
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, size=k, mode="nearest")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--figs", default="fase1_figs")
    a, _ = ap.parse_known_args()
    os.makedirs(a.figs, exist_ok=True)

    tr = np.load(find("train.npz", a.data))
    te = np.load(find("test.npz", a.data))
    st = np.load(find("static.npz", a.data))
    sub_path = find("sample_submission.csv", a.data)

    # ------------------------------------------------------------------ 1. inventario
    hdr("1. INVENTARIO DE ARCHIVOS")
    for nm, f in [("train", tr), ("test", te), ("static", st)]:
        for k in f.files:
            v = f[k]
            print(f"{nm:7s} {k:12s} shape={str(v.shape):22s} dtype={v.dtype}")
    Xtr, Ytr = tr["X"].astype(np.float32), tr["Y"].astype(np.float32)
    ttr, tte = tr["times"].astype(str), te["times"].astype(str)
    Xte = te["X"].astype(np.float32)
    dem = st["dem"].astype(np.float32)
    glat, glon, wlat, wlon = (st[k].astype(np.float64) for k in ["gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons"])

    # ------------------------------------------------------------------ 2. tiempo
    hdr("2. EJE TEMPORAL")
    for nm, t in [("train", ttr), ("test", tte)]:
        ts = pd.to_datetime(t, format="%Y%m%d%H")
        d = np.diff(ts.values).astype("timedelta64[h]").astype(int)
        full = int((ts[-1] - ts[0]) / pd.Timedelta(hours=1)) + 1
        print(f"{nm}: {len(t)} horas | {t[0]} -> {t[-1]} | esperadas si fuera horario completo: {full} "
              f"| faltan {full - len(t)} | ordenado={bool((d > 0).all())} | duplicados={len(t) - len(set(t))}")
        gaps = [(str(ts[i]), int(g - 1)) for i, g in enumerate(d) if g > 1]
        print(f"   huecos: {len(gaps)}  (los 10 más largos, [inicio, horas faltantes]): "
              f"{sorted(gaps, key=lambda x: -x[1])[:10]}")
        hc = pd.Series(hour(t)).value_counts().sort_index()
        print(f"   horas UTC presentes: {hc.index.tolist()}  (min/max conteo por hora: {hc.min()}/{hc.max()})")

    sample = pd.read_csv(sub_path)
    parts = sample["id"].str.split("_", expand=True)
    ev_t = np.unique(parts[0].values)
    print(f"\nsample_submission: {sample.shape} columnas={list(sample.columns)}")
    print(f"   horas evaluadas: {len(ev_t)} | horas UTC: {sorted(set(int(x[-2:]) for x in ev_t))}")
    print(f"   filas = horas_eval x 2450 ? {len(ev_t) * NPIX} vs {len(sample)}")
    print(f"   filas/cols: r∈[{parts[1].astype(int).min()},{parts[1].astype(int).max()}] "
          f"c∈[{parts[2].astype(int).min()},{parts[2].astype(int).max()}]")
    print(f"   ¿todas las horas evaluadas están en test.npz? {set(ev_t) <= set(tte)}")
    print(f"   horas de test.npz NO evaluadas (usables solo como input de contexto): {len(set(tte) - set(ev_t))}")
    ts_te = pd.to_datetime(tte, format="%Y%m%d%H"); S = set(ts_te)
    for lag in [-2, -1, 1, 2]:
        ok = np.mean([(pd.Timestamp(pd.to_datetime(k, format='%Y%m%d%H')) + pd.Timedelta(hours=lag)) in S for k in ev_t])
        print(f"   fracción de horas evaluadas con GFS disponible en t{lag:+d}: {ok:.3f}")
    clim_cols = sample[VARS].describe().loc[["mean", "min", "max"]].round(3)
    print("   valores del sample (climatología oficial):\n" + clim_cols.to_string())

    # ------------------------------------------------------------------ 3. calidad
    hdr("3. CALIDAD Y RANGOS FÍSICOS (p0.1 / p50 / p99.9)")
    for nm, A in [("X train (GFS)", Xtr), ("X test  (GFS)", Xte), ("Y train (WRF)", Ytr)]:
        print(nm)
        for k, v in enumerate(VARS):
            x = A[..., k]
            nbad = int((~np.isfinite(x)).sum())
            q = np.nanpercentile(x, [0.1, 50, 99.9])
            extra = ""
            if v == "RH":
                extra = f" | <0: {int((x < 0).sum())}  >100: {int((x > 100).sum())}  ==100: {np.mean(x >= 99.999):.3%}"
            print(f"   {v:2s} [{UNITS[k]:3s}] NaN/Inf={nbad:<6d} min={np.nanmin(x):9.3f} "
                  f"p={q.round(3)} max={np.nanmax(x):9.3f}{extra}")
    print(f"DEM: {dem.shape} min={dem.min():.0f} m  max={dem.max():.0f} m  media={dem.mean():.0f} m  NaN={int(np.isnan(dem).sum())}")

    # ------------------------------------------------------------------ 4. orientación
    hdr("4. GEOMETRÍA Y ORIENTACIÓN")
    print(f"gfs_lats {glat.shape}: {glat[0]:.3f} -> {glat[-1]:.3f} (paso {np.diff(glat).mean():+.4f})  "
          f"=> fila 0 = {'NORTE' if glat[0] > glat[-1] else 'SUR'}")
    print(f"gfs_lons {glon.shape}: {glon[0]:.3f} -> {glon[-1]:.3f} (paso {np.diff(glon).mean():+.4f})")
    print(f"wrf_lats {wlat.shape}: {wlat[0]:.3f} -> {wlat[-1]:.3f} (paso {np.diff(wlat).mean():+.4f})  "
          f"=> fila 0 = {'NORTE' if wlat[0] > wlat[-1] else 'SUR'}")
    print(f"wrf_lons {wlon.shape}: {wlon[0]:.3f} -> {wlon[-1]:.3f} (paso {np.diff(wlon).mean():+.4f})")
    inside = (min(glat) <= wlat.min() and wlat.max() <= max(glat) and glon.min() <= wlon.min() and wlon.max() <= glon.max())
    print(f"WRF dentro de GFS: {inside} | margen GFS alrededor del WRF (grados): "
          f"N {max(glat) - wlat.max():.2f}, S {wlat.min() - min(glat):.2f}, "
          f"O {wlon.min() - glon.min():.2f}, E {glon.max() - wlon.max():.2f}")
    # chequeo de sanidad: el DEM más alto debería coincidir con T WRF más fría (si fila0=sur en ambos)
    Tm = Ytr[..., 1].mean(0)
    print(f"corr espacial( DEM , T_WRF media ) = {np.corrcoef(dem.ravel(), Tm.ravel())[0, 1]:+.3f}  "
          f"(debe ser fuertemente NEGATIVA; si es ~0 hay un flip de ejes)")
    print(f"corr con DEM volteado (flipud)    = {np.corrcoef(np.flipud(dem).ravel(), Tm.ravel())[0, 1]:+.3f}")

    # ------------------------------------------------------------------ 5. baselines numéricos
    hdr("5. BASELINES SIN APRENDIZAJE (validación = 16–30 jun, solo horas 00,03,...,21)")
    W = bilinear_weights(glat, glon, wlat, wlon)
    Itr = interp(Xtr, W)
    h_tr = hour(ttr)
    val = (ttr >= "2024061600"); trn = ~val
    ev = val & (h_tr % 3 == 0)
    clim = np.stack([Ytr[trn & (h_tr == k)].mean(0) if (trn & (h_tr == k)).any() else np.full((70, 35, 4), np.nan)
                     for k in range(24)])
    r_clim = rmse(clim[h_tr[ev]], Ytr[ev])
    print(f"Climatología horaria (fit en train)  RMSE: {fmt(r_clim)}")
    print(f"   score con constantes OFICIALES: {score(r_clim, CONST):+.4f}   (en test, por definición = 0)")
    print(f"   ratio RMSE_clim_val / constante oficial: {fmt(r_clim / CONST)}")
    print("   -> si estos ratios ≠ 1, la escala de tu CV local no es la del LB; reporta ambos scores.")

    r_int = rmse(Itr[ev], Ytr[ev])
    print(f"GFS interpolado bilineal (crudo)      RMSE: {fmt(r_int)} | score_oficial={score(r_int, CONST):+.4f} "
          f"| score_local={score(r_int, r_clim):+.4f}")
    dz = dem - smooth(dem, 9)                           # relieve sub-celda que GFS no 've'
    P = Itr.copy(); P[..., 1] -= 0.0065 * dz
    r_lr = rmse(P[ev], Ytr[ev])
    print(f"+ corrección lapse rate 6.5 K/km en T RMSE: {fmt(r_lr)} | score_oficial={score(r_lr, CONST):+.4f}")
    # sesgo medio por píxel y hora estimado en train (= 'MOS' trivial)
    bias = np.stack([(Ytr[trn & (h_tr == k)] - Itr[trn & (h_tr == k)]).mean(0) for k in range(24)])
    P2 = Itr + bias[h_tr]
    r_mos = rmse(P2[ev], Ytr[ev])
    print(f"GFS interp + sesgo(píxel,hora) de train RMSE: {fmt(r_mos)} | score_oficial={score(r_mos, CONST):+.4f} "
          f"| score_local={score(r_mos, r_clim):+.4f}")
    print("   (este último es el baseline 'físico-estadístico' a batir; el starter CNN reporta ≈0.40 en LB)")

    # ------------------------------------------------------------------ 6. física
    hdr("6. EDA FÍSICO")
    print("Pendiente espacial T vs DEM por hora UTC (media temporal en train), K/km, y R²:")
    rows = []
    for k in sorted(set(h_tr)):
        m = (h_tr == k)
        y = Ytr[m, ..., 1].mean(0).ravel(); x = dem.ravel() / 1000
        b, c = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
        yr = Ytr[m, ..., 0].mean(0).ravel(); br, _ = np.polyfit(x, yr, 1)
        rows.append((k, (k - 5) % 24, b, r2, br))
    print(pd.DataFrame(rows, columns=["UTC", "hora_Lima", "dT/dz K/km", "R2_T", "dRH/dz %/km"]).round(3).to_string(index=False))

    print("\nSesgo medio de GFS interp vs WRF (WRF - GFS) por variable, por hora UTC (promedio de dominio):")
    bb = pd.DataFrame(bias.mean((1, 2)), columns=VARS); bb.index.name = "UTC"
    print(bb.loc[sorted(set(h_tr))].round(3).to_string())

    print("\nDeriva estacional (media de dominio por mes) — ¿qué tan distinto será jul–ago?")
    mon_tr = pd.to_datetime(ttr, format="%Y%m%d%H").month
    mon_te = pd.to_datetime(tte, format="%Y%m%d%H").month
    rows = []
    for m in sorted(set(mon_tr)):
        s = mon_tr == m
        rows.append(["train", m] + list(Xtr[s].mean((0, 1, 2))) + list(Ytr[s].mean((0, 1, 2))))
    for m in sorted(set(mon_te)):
        s = mon_te == m
        rows.append(["test", m] + list(Xte[s].mean((0, 1, 2))) + [np.nan] * 4)
    cols = ["set", "mes"] + [f"GFS_{v}" for v in VARS] + [f"WRF_{v}" for v in VARS]
    print(pd.DataFrame(rows, columns=cols).round(2).to_string(index=False))

    print("\nAutocorrelación temporal del residuo (WRF - GFSinterp - sesgo), media de dominio y por píxel:")
    R = Ytr - P2                                       # residuo tras el baseline MOS
    tsr = pd.to_datetime(ttr, format="%Y%m%d%H"); pos = {k: i for i, k in enumerate(tsr)}
    for lag in [1, 3, 6, 12, 24, 72]:
        pairs = [(i, pos[k + pd.Timedelta(hours=lag)]) for i, k in enumerate(tsr) if (k + pd.Timedelta(hours=lag)) in pos]
        a_, b_ = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        out = []
        for k in range(4):
            r1 = R[a_, ..., k].reshape(len(a_), -1); r2 = R[b_, ..., k].reshape(len(b_), -1)
            r1c = r1 - r1.mean(0); r2c = r2 - r2.mean(0)
            out.append(((r1c * r2c).mean(0) / (r1c.std(0) * r2c.std(0) + 1e-9)).mean())
        print(f"   lag {lag:3d} h: " + "  ".join(f"{v}={o:+.2f}" for v, o in zip(VARS, out)))
    print("   -> alta autocorrelación a 24–72 h = el WRF tiene 'memoria' que GFS(t-2..t) no explica"
          " (y obliga a dejar un embargo entre train y val).")

    # varianza explicada por píxel (R²) del baseline MOS, por variable, en val
    print("\nR² por píxel del baseline MOS en val (percentiles 10/50/90) — dónde sobra señal por aprender:")
    for k, v in enumerate(VARS):
        y = Ytr[ev, ..., k]; p = P2[ev, ..., k]
        r2 = 1 - ((y - p) ** 2).mean(0) / (y.var(0) + 1e-9)
        print(f"   {v}: {np.percentile(r2, [10, 50, 90]).round(3)}")

    # ------------------------------------------------------------------ 7. figuras
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ext = [wlon.min(), wlon.max(), wlat.min(), wlat.max()]
        fig, ax = plt.subplots(1, 5, figsize=(20, 7))
        im = ax[0].imshow(dem, origin="lower", extent=ext, cmap="terrain"); ax[0].set_title("DEM (m)"); fig.colorbar(im, ax=ax[0], shrink=.6)
        for k, v in enumerate(VARS):
            e = np.sqrt(((P2[ev, ..., k] - Ytr[ev, ..., k]) ** 2).mean(0))
            im = ax[k + 1].imshow(e, origin="lower", extent=ext, cmap="magma"); ax[k + 1].set_title(f"RMSE MOS val · {v}")
            fig.colorbar(im, ax=ax[k + 1], shrink=.6)
        fig.tight_layout(); fig.savefig(f"{a.figs}/rmse_mapas_baseline.png", dpi=110); plt.close(fig)

        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        for k, v in enumerate(VARS):
            yw = [Ytr[h_tr == h, ..., k].mean() for h in range(24) if (h_tr == h).any()]
            yg = [Itr[h_tr == h, ..., k].mean() for h in range(24) if (h_tr == h).any()]
            hh = [h for h in range(24) if (h_tr == h).any()]
            ax[k].plot(hh, yw, "o-", label="WRF"); ax[k].plot(hh, yg, "s--", label="GFS interp")
            ax[k].set_title(f"Ciclo diurno · {v} ({UNITS[k]})"); ax[k].set_xlabel("hora UTC (Lima = UTC-5)"); ax[k].legend()
        fig.tight_layout(); fig.savefig(f"{a.figs}/ciclo_diurno.png", dpi=110); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for j, h in enumerate([9, 18]):
            m = h_tr == h
            if m.any():
                ax[j].scatter(dem.ravel(), Ytr[m, ..., 1].mean(0).ravel() - 273.15, s=3, alpha=.5, label="WRF")
                ax[j].scatter(dem.ravel(), Itr[m, ..., 1].mean(0).ravel() - 273.15, s=3, alpha=.5, label="GFS interp")
                ax[j].set_title(f"T vs elevación · {h} UTC ({(h - 5) % 24} h Lima)"); ax[j].set_xlabel("DEM (m)"); ax[j].set_ylabel("°C"); ax[j].legend()
        fig.tight_layout(); fig.savefig(f"{a.figs}/T_vs_dem.png", dpi=110); plt.close(fig)
        print(f"\nFiguras guardadas en {a.figs}/")
    except Exception as e:
        print("No se pudieron generar figuras:", e)


if __name__ == "__main__":
    main()