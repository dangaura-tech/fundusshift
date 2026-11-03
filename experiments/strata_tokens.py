"""STRATA gate 2: optic-disc location and angular Fourier magnitudes of encoder tokens around it.
Magnitudes do not change when the image is rotated about the disc or flipped, so no eye side or fovea is needed.
python -m experiments.strata_tokens --bench data/bench --images data/processed --out data/strata
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RADII = (1.5, 3.0, 5.0)  # outer edge of each ring, in disc radii
DISC = 22 / 224          # disc radius as a fraction of the image width
SECTORS = 12
MODES = 4


def locate_disc(rgb):
    """(x, y) of the disc on a 224 px image, with no disc labels: brightest smooth blob of red + green inside the field of view."""
    import cv2
    a = rgb.astype(np.float32)
    mask = cv2.erode((a.sum(2) > 30).astype(np.uint8), np.ones((15, 15), np.uint8))
    s = cv2.GaussianBlur(a[..., 0] + a[..., 1], (0, 0), 9) * mask
    y, x = np.unravel_index(int(s.argmax()), s.shape)
    return float(x), float(y)


def spectrum(tok, centre):
    """tok: (B, G, G, D); centre: (B, 2) as fractions of the width. Returns (B, rings, modes, D)."""
    B, G, _, D = tok.shape
    xs = (torch.arange(G, device=tok.device, dtype=tok.dtype) + .5) / G
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    out = []
    for b in range(B):
        dx, dy = xx - centre[b, 0], yy - centre[b, 1]
        r = (dx ** 2 + dy ** 2).sqrt() / DISC
        sec = (((torch.atan2(dy, dx) + np.pi) / (2 * np.pi)) * SECTORS).long().clamp_max(SECTORS - 1)
        ring = torch.bucketize(r, torch.tensor(RADII, device=tok.device, dtype=tok.dtype))
        V = torch.zeros(len(RADII), SECTORS, D, device=tok.device, dtype=tok.dtype)
        for i in range(len(RADII)):
            for s in range(SECTORS):
                m = (ring == i) & (sec == s)
                if m.any():
                    V[i, s] = tok[b][m].mean(0)
        out.append(torch.fft.rfft(V, dim=1).abs()[:, :MODES])
    return torch.stack(out)


class Transform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, im):
        x, y = locate_disc(np.asarray(im.resize((224, 224), Image.BILINEAR)))
        return self.processor(images=im, return_tensors="pt")["pixel_values"][0], torch.tensor([x / 224, y / 224], dtype=torch.float32)


def main(argv=None):
    from .data import ImageTable, load_table
    from .followup import Medsiglip
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t, _ = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tf = Transform(sg.proc.image_processor)
    f, fc, fd = out / "spectrum_medsiglip.npy", out / "disc.npy", out / "done.npy"
    S = C = None
    done = int(np.load(fd)) if fd.exists() else 0
    bs = 32
    dl = torch.utils.data.DataLoader(ImageTable(t.iloc[done:], a.images, tf), bs, num_workers=4)
    with torch.no_grad():
        for n, ((x, c), _) in enumerate(dl):
            h = sg.m.vision_model(pixel_values=x.to(dev)).last_hidden_state
            G = int(h.shape[1] ** .5)
            sp = spectrum(h.reshape(len(x), G, G, -1), c.to(dev)).cpu().numpy().astype(np.float16)
            if S is None:
                shape = (len(t),) + sp.shape[1:]
                mode = "r+" if done else "w+"
                S = np.lib.format.open_memmap(f, mode=mode, dtype=np.float16, shape=shape)
                C = np.lib.format.open_memmap(fc, mode=mode, dtype=np.float32, shape=(len(t), 2))
            S[done:done + len(x)], C[done:done + len(x)] = sp, c.numpy()
            done += len(x)
            if n % 50 == 0:
                S.flush(); C.flush(); np.save(fd, done)
                print(done, len(t), flush=True)
    S.flush(); C.flush(); np.save(fd, done)
    print("[ok]", done, flush=True)


if __name__ == "__main__":
    main()
