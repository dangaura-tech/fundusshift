"""Training losses, post-hoc calibrators and evaluation metrics used by every FundusShift experiment.

Losses (train-time calibration)
  ce      cross-entropy
  ls      label smoothing, eps = 0.1                                  (Szegedy et al. 2016; Müller et al. 2019)
  focal   focal loss, gamma = 3                                       (Lin et al. 2017; Mukhoti et al. 2020)
  flsd    sample-dependent focal, gamma = 5 if p_y < 0.2 else 3 (FLSD-53)   (Mukhoti et al. 2020)
  mdca    CE + beta * MDCA, beta = 1                                  (Hebbalaguppe et al., CVPR 2022)
  mbls    CE + lambda * sum_j relu(max(z) - z_j - m), m = 10, lambda = 0.1   (Liu et al., CVPR 2022)
Post-hoc calibrators, fitted on calibration cal_A only
  none    raw softmax
  ts      temperature scaling, T by NLL                               (Guo et al. 2017)
  fts     focal temperature scaling: softmax(z / T) -> focal calibration map(gamma); (gamma, T) by NLL on a grid
                                                                      (Komisarenko & Kull, ECAI 2024)
Metrics: accuracy, balanced accuracy, macro-F1, macro AUROC, NLL, Brier, ECE (15 equal-width bins), adaptive ECE
(15 equal-mass bins), classwise ECE, MCE, AURC / E-AURC, selective risk at fixed coverage, open-set AUROC.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

LOSSES = ["ce", "ls", "focal", "flsd", "mdca", "mbls"]
POSTHOC = ["none", "ts", "fts"]
LOG_T = (np.log(1e-3), np.log(1e3))     # searched in log T: zero-shot logits carry a scale near 100


# ------------------------------------------------------------------------------------------------ losses
def focal_loss(logits, y, gamma):
    logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    p = logp.exp()
    if callable(gamma):
        gamma = gamma(p.detach())
    return (-(1 - p) ** gamma * logp).mean()


def mdca(logits, y):
    """(1/K) sum_j | mean_i p_ij - mean_i 1[y_i = j] |"""
    p = F.softmax(logits, 1)
    onehot = F.one_hot(y, logits.shape[1]).float()
    return (p.mean(0) - onehot.mean(0)).abs().mean()


def loss_fn(name):
    def f(logits, y):
        if name == "ce":
            return F.cross_entropy(logits, y)
        if name == "ls":
            return F.cross_entropy(logits, y, label_smoothing=0.1)
        if name == "focal":
            return focal_loss(logits, y, 3.0)
        if name == "flsd":
            return focal_loss(logits, y, lambda p: torch.where(p < 0.2, 5.0, 3.0))
        if name == "mdca":
            return F.cross_entropy(logits, y) + 1.0 * mdca(logits, y)
        if name == "mbls":
            gap = logits.max(1, keepdim=True).values - logits
            return F.cross_entropy(logits, y) + 0.1 * F.relu(gap - 10.0).sum(1).mean()
        raise ValueError(name)
    return f


# ------------------------------------------------------------------------------------------- post-hoc
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def focal_map(q, gamma):
    """Multiclass focal calibration map (Komisarenko & Kull 2024, Prop. 4); gamma = 0 is the identity."""
    if abs(gamma) < 1e-9:
        return q
    q = np.clip(q, 1e-7, 1 - 1e-7)
    t = 1.0 / ((1 - q) ** gamma * (gamma * np.log(q) / (1 - q) - 1.0 / q))
    return t / t.sum(1, keepdims=True)


class Calibrator:
    def __init__(self, kind):
        self.kind, self.T, self.gamma = kind, 1.0, 0.0

    def fit(self, z, y):
        if self.kind == "ts":
            self.T = float(np.exp(minimize_scalar(lambda u: nll(softmax(z / np.exp(u)), y), bounds=LOG_T, method="bounded").x))
        elif self.kind == "fts":
            best = (np.inf, 1.0, 0.0)
            for g in [-0.5, -0.25, 0.0, 0.05, 0.25, 0.37, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
                r = minimize_scalar(lambda u: nll(focal_map(softmax(z / np.exp(u)), g), y), bounds=LOG_T, method="bounded")
                if r.fun < best[0]:
                    best = (r.fun, float(np.exp(r.x)), g)
            _, self.T, self.gamma = best
        return self

    def __call__(self, z):
        if self.kind == "none":
            return softmax(z)
        p = softmax(z / self.T)
        return focal_map(p, self.gamma) if self.kind == "fts" else p


# -------------------------------------------------------------------------------------------- metrics
def ece(p, y, bins=15, adaptive=False):
    conf, pred = p.max(1), p.argmax(1)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        edges[0] = np.nextafter(edges[0], -np.inf)     # bins are (lo, hi]: keep the least confident sample
    else:
        edges = np.linspace(0, 1, bins + 1)
    e, mce = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs((pred[m] == y[m]).mean() - conf[m].mean())
            e += m.mean() * gap
            mce = max(mce, gap)
    return float(e), float(mce)


def classwise_ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for k in range(p.shape[1]):
        pk, yk = p[:, k], (y == k)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pk > lo) & (pk <= hi)
            if m.any():
                tot += m.mean() * abs(yk[m].mean() - pk[m].mean())
    return float(tot / p.shape[1])


def selective(p, y, coverages=(0.8, 0.9)):
    conf, err = p.max(1), (p.argmax(1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    aurc = float(risk.mean())
    k = np.arange(1, len(y) + 1)
    opt = float((np.clip(k - (len(y) - err.sum()), 0, None) / k).mean())   # all correct predictions ranked first
    out = {"aurc": aurc, "e_aurc": aurc - opt}
    for c in coverages:
        out[f"risk@{int(c * 100)}"] = float(risk[max(int(c * len(y)) - 1, 0)])
    return out


def classification_metrics(p, y, source=None):
    pred = p.argmax(1)
    k = p.shape[1]
    present = sorted(set(y))
    out = {"n": int(len(y)), "acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=present, zero_division=0)),
           "nll": nll(p, y), "brier": float(((p - np.eye(k)[y]) ** 2).sum(1).mean())}
    try:
        out["auroc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(k)))) \
            if len(present) == k else float("nan")
    except ValueError:
        out["auroc"] = float("nan")
    out["ece"], out["mce"] = ece(p, y)
    out["ace"], _ = ece(p, y, adaptive=True)
    out["cw_ece"] = classwise_ece(p, y)
    out.update(selective(p, y))
    if source is not None:
        per = {s: float(balanced_accuracy_score(y[source == s], pred[source == s])) for s in np.unique(source)}
        out["per_source_bacc"] = per
        out["worst_source_bacc"] = min(per.values())
        out["mean_source_bacc"] = float(np.mean(list(per.values())))
        out["per_source_ece"] = {s: ece(p[source == s], y[source == s])[0] for s in np.unique(source)}
    return out


def openset_auroc(p_in, p_out):
    """Referral of out-of-label-space images: score = 1 - max prob (MSP) and predictive entropy."""
    lab = np.r_[np.zeros(len(p_in)), np.ones(len(p_out))]
    msp = np.r_[1 - p_in.max(1), 1 - p_out.max(1)]
    ent = np.r_[-(p_in * np.log(p_in + 1e-12)).sum(1), -(p_out * np.log(p_out + 1e-12)).sum(1)]
    return {"auroc_msp": float(roc_auc_score(lab, msp)), "auroc_entropy": float(roc_auc_score(lab, ent))}


def bootstrap_ci(fn, groups, n_boot=1000, seed=0):
    """95% percentile CI of fn(index_array) resampling leakage groups (patients / duplicate groups) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(fn(np.concatenate([members[g] for g in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
