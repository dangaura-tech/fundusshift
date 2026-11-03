"""Fundus-specific near-duplicate detection.

Generic perceptual hashes (pHash/dHash) and generic SSL embeddings (DINOv2 CLS) do not separate a re-encoded copy
of a fundus photograph from a different eye: every fundus is an orange disc with a darker macula. What is unique to
an eye is its vessel tree, so:
  1. candidate signature: green channel, band-pass (difference of Gaussians), 32x32, unit norm  -> cosine
  2. verification: illumination-normalised band-pass at 256 px inside the *eroded* field of view, normalised
     cross-correlation maximised over small shifts (+-4 px)
The field-of-view border, camera notches and oval apertures are shared by every image of a camera model; with a plain
circular mask they dominated the correlation of unrelated eyes (DDR, RFMiD 2.0), hence the eroded FOV mask.
"""
import numpy as np
from PIL import Image, ImageFilter

from .hashing import crop_fundus

SIG = 32


def _bandpass(img, size, r_small, r_big):
    g = crop_fundus(img.convert("RGB")).split()[1].resize((size, size), Image.BILINEAR)
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(r_small)), np.float32)
    b = np.asarray(g.filter(ImageFilter.GaussianBlur(r_big)), np.float32)
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - size / 2 + .5) ** 2 + (xx - size / 2 + .5) ** 2) < (0.42 * size) ** 2
    return (a - b) * mask, mask


def signature(img):
    d, _ = _bandpass(img, SIG * 4, 2.4, 12)
    d = d.reshape(SIG, 4, SIG, 4).mean((1, 3)).ravel()
    d -= d.mean()
    return (d / (np.linalg.norm(d) + 1e-9)).astype(np.float16)


def fine_map(img, size=256, erode=0.08):
    rgb = crop_fundus(img.convert("RGB")).resize((size, size), Image.BILINEAR)
    fov = Image.fromarray(((np.asarray(rgb).max(axis=2) > 20) * 255).astype(np.uint8))
    k = min(2 * int(erode * size) + 1, 61)
    fov = np.asarray(fov.filter(ImageFilter.MinFilter(k))) > 0
    g = rgb.split()[1]
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(1.0)), np.float32)
    b = np.asarray(g.filter(ImageFilter.GaussianBlur(6)), np.float32)
    yy, xx = np.mgrid[:size, :size]
    mask = fov & (((yy - size / 2 + .5) ** 2 + (xx - size / 2 + .5) ** 2) < (0.40 * size) ** 2)
    return (a - b) / (b + 8.0) * mask, mask


def verify(a, b, max_shift=4):
    """Max normalised cross-correlation of two fine maps over integer shifts, on the joint FOV mask."""
    (da, ma), (db, mb) = a, b
    s, n = max_shift, da.shape[0]
    best = -1.0
    for dy in range(-s, s + 1, 2):
        for dx in range(-s, s + 1, 2):
            M = ma[s:n - s, s:n - s] & mb[s + dy:n - s + dy, s + dx:n - s + dx]
            if M.sum() < 500:
                continue
            x = da[s:n - s, s:n - s][M]
            y = db[s + dy:n - s + dy, s + dx:n - s + dx][M]
            x = (x - x.mean()) / (x.std() + 1e-6)
            y = (y - y.mean()) / (y.std() + 1e-6)
            best = max(best, float((x * y).mean()))
    return best


def pairs_above(S, thr, block=4096):
    """All (i, j), i < j with S_i . S_j >= thr (S: n x d unit vectors)."""
    S = S.astype(np.float32)
    out = []
    for i0 in range(0, len(S), block):
        sim = S[i0:i0 + block] @ S.T
        ii, jj = np.nonzero(sim >= thr)
        keep = jj > ii + i0
        out.extend(zip((ii[keep] + i0).tolist(), jj[keep].tolist(), sim[ii[keep], jj[keep]].tolist()))
    return out
