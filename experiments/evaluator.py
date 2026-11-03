"""One evaluator for every predictor: logits for all rows of the split table -> raw / TS / FTS records."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .calib import POSTHOC, Calibrator, bootstrap_ci, classification_metrics, ece, openset_auroc


def referral_threshold(p, y, target=0.05):
    conf, err = p.max(1), p.argmax(1) != y
    order = np.argsort(-conf)
    risk = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(risk <= target)
    return float(conf[order][ok.max()]) if len(ok) else 1.0


def evaluate(Z, t, meta, n_boot=300):
    sp, y, src, grp = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy()
    ca = (sp == "calibration") & (t.cal_subset == "cal_A").to_numpy()
    cb = (sp == "calibration") & (t.cal_subset == "cal_B").to_numpy()
    rows = []
    for ph in POSTHOC:
        cal = Calibrator(ph).fit(Z[ca], y[ca])
        P = cal(Z)
        thr = referral_threshold(P[cb], y[cb])
        rec = {**meta, "posthoc": ph, "T": cal.T, "gamma_fts": cal.gamma, "referral_threshold": thr}
        for s in ["validation", "test", "test_ood"]:
            m = sp == s
            if not m.any():
                continue
            r = classification_metrics(P[m], y[m], src[m])
            acc = P[m].max(1) >= thr
            r["coverage_at_thr"] = float(acc.mean())
            r["risk_at_thr"] = float((P[m][acc].argmax(1) != y[m][acc]).mean()) if acc.any() else None
            if n_boot and s != "validation":
                Pm, ym, gm = P[m], y[m], grp[m]
                r["bacc_ci"] = bootstrap_ci(lambda i: balanced_accuracy_score(ym[i], Pm[i].argmax(1)), gm, n_boot)
                r["ece_ci"] = bootstrap_ci(lambda i: ece(Pm[i], ym[i])[0], gm, n_boot)
            rec[s] = r
        if (sp == "test").any() and (sp == "test_openset").any():
            rec["openset"] = openset_auroc(P[sp == "test"], P[sp == "test_openset"])
        rows.append(rec)
    return rows
