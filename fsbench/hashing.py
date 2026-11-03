"""Exact (SHA-256) and perceptual (DCT pHash on the cropped fundus disc) image fingerprints."""
import hashlib

import numpy as np
from PIL import Image

_N = 32
_DCT = np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def crop_fundus(img, thresh=15):
    """Crop to the bounding box of the illuminated fundus so black borders / padding don't dominate the hash."""
    a = np.asarray(img.convert("RGB")).max(axis=2)
    ys, xs = np.nonzero(a > thresh)
    if len(xs) < 100:
        return img
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def phash(img):
    g = crop_fundus(img).convert("L").resize((_N, _N), Image.BILINEAR)
    d = _DCT @ np.asarray(g, dtype=np.float64) @ _DCT.T
    low = d[:8, :8].flatten()[1:]          # drop DC term
    bits = np.concatenate([[False], low > np.median(low)])
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(img):
    g = np.asarray(crop_fundus(img).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fingerprint(path):
    """Return dict with sha256, phash/dhash (hex), width, height, or an error string."""
    try:
        sha = sha256_file(path)
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            ph, dh = phash(im), dhash(im)
        return {"sha256": sha, "phash": f"{ph:016x}", "dhash": f"{dh:016x}", "width": w, "height": h, "read_error": ""}
    except Exception as e:  # corrupt / unreadable file is recorded, not fatal
        return {"sha256": "", "phash": "", "dhash": "", "width": 0, "height": 0, "read_error": repr(e)[:200]}


def near_duplicate_pairs(hashes, max_dist=6, max_bucket=5000):
    """All index pairs with Hamming(phash) <= max_dist (max_dist <= 7).

    Pigeonhole: split 64 bits into 8 bytes; two hashes within distance 7 share at least one byte exactly,
    so candidates come from equal-byte buckets. Huge buckets (e.g. blank images) are skipped and reported.
    """
    assert max_dist <= 7
    h = np.array([int(x, 16) for x in hashes], dtype=np.uint64)
    pairs, skipped = set(), 0
    for band in range(8):
        key = (h >> np.uint64(8 * band)) & np.uint64(0xFF)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, bounds):
            if len(grp) < 2:
                continue
            if len(grp) > max_bucket:
                skipped += 1
                continue
            x = h[grp]
            for s in range(0, len(grp), 512):      # row chunks keep memory bounded
                dist = np.bitwise_count(x[s:s + 512, None] ^ x[None, :])
                ii, jj = np.nonzero(dist <= max_dist)
                for i, j in zip(grp[s + ii], grp[jj]):
                    if i != j:
                        pairs.add((int(min(i, j)), int(max(i, j))))
    return sorted(pairs), skipped
