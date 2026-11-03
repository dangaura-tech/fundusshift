"""16-shot CoOp (Zhou et al., IJCV 2022): 16 learnable context tokens + class name, image tower frozen.

Shots are sampled per class from train (one draw per seed, one image per leakage group). Text tower frozen; only the
context is optimised (SGD lr 0.002, cosine, 200 steps of full-batch shots). 3 seeds, plus their logit-average ensemble.
python -m experiments.coop --bench data/bench --features runs/features --out runs/coop
Same recipe, ablations: --losses ce focal ls mbls mdca zs_penalty zs_norm (ZS terms: Murugesan et al., ECCV 2024, against the frozen
zero-shot CLIP-ensemble logits of the same images); --ctx-init "a photo of a" (4 context tokens, CoOp's published start);
--lp (CoOp-LP: then a linear head starting at the learned prompts, 200 steps, lr 1e-3).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .backbones import VLM, build_clip
from .calib import loss_fn
from .data import load_table
from .evaluator import evaluate
from .prompts import NAMES
from .zeroshot_clip import text_weights

N_CTX = 16


class Prompter(torch.nn.Module):
    """Runs a text tower on token embeddings with a learnable context spliced over placeholder tokens."""

    def __init__(self, model, tok, classes, dev, ctx_init=None):
        super().__init__()
        self.text = model.text if hasattr(model, "text") else model
        self.model = model
        base = tok([NAMES[c] + "." for c in classes]).to(dev)
        pre = 0                                  # special tokens before the text = common prefix over the classes
        while (base[:, pre] == base[0, pre]).all():
            pre += 1
        L = base.shape[1]
        init_ids = tok([ctx_init]).to(dev)[0, pre:pre + 4] if ctx_init else None
        n_ctx = 4 if ctx_init else N_CTX
        ph = base[0, pre]                        # placeholder id; every placeholder embedding is replaced by the context
        ids = torch.cat([base[:, :pre], ph.repeat(len(base), n_ctx), base[:, pre:]], 1)[:, :L]
        self.ids = ids
        self.mask = torch.zeros_like(ids, dtype=torch.bool)
        self.mask[:, pre:pre + n_ctx] = True
        self.hf = hasattr(self.text, "transformer") and hasattr(self.text.transformer, "embeddings")
        emb = self.embed_tokens()
        start = emb(init_ids).detach().float().clone() if ctx_init else torch.normal(0, 0.02, (N_CTX, emb(ids).shape[-1]), device=dev)
        self.ctx = torch.nn.Parameter(start)

    def embed_tokens(self):
        if self.hf:
            return self.text.transformer.get_input_embeddings()
        return self.text.token_embedding if hasattr(self.text, "token_embedding") else self.model.token_embedding

    def forward(self):
        emb = self.embed_tokens()(self.ids).clone()
        emb[self.mask] = self.ctx.repeat(len(self.ids), 1).to(emb.dtype)
        if self.hf:
            out = self.text.transformer(inputs_embeds=emb, attention_mask=(self.ids != self.text.config.pad_token_id).long())
            pooled = self.text.pooler(out, self.ids) if hasattr(self.text, "pooler") else out.last_hidden_state[:, 0]
            return self.text.proj(pooled)
        t = self.text
        x = emb + t.positional_embedding[:emb.shape[1]].to(emb.dtype)
        attn = getattr(t, "attn_mask", None)
        x = t.transformer(x, attn_mask=attn)
        x = t.ln_final(x)
        pool = getattr(t, "pool_type", "argmax")
        if pool == "last":
            pooled = x[:, -1]
        elif pool == "first":
            pooled = x[:, 0]
        else:
            pooled = x[torch.arange(len(x)), self.ids.argmax(dim=-1)]
        return t.text_projection(pooled) if isinstance(t.text_projection, torch.nn.Module) else pooled @ t.text_projection


def sample_shots(t, seed, k=16):
    rng = np.random.default_rng(seed)
    tr = t[t.split == "train"]
    out = []
    for y in sorted(tr.y.unique()):
        groups = tr[tr.y == y].groupby("group_id").head(1)
        out += rng.choice(groups.index.to_numpy(), k, replace=False).tolist()
    return np.array(out)


def train_loss(name, z, y, zs):
    """z: CoOp logits; zs: zero-shot logits of the same images (constants), used by the two ZS losses."""
    if name == "zs_penalty":
        hi, lo = zs.max(1, keepdim=True).values, zs.min(1, keepdim=True).values
        return F.cross_entropy(z, y) + 10 * (F.relu(z - hi).mean() + F.relu(lo - z).mean())
    if name == "zs_norm":
        lo, hi = z.min(1, keepdim=True).values, z.max(1, keepdim=True).values
        zlo, zhi = zs.min(1, keepdim=True).values, zs.max(1, keepdim=True).values
        return F.cross_entropy((z - lo) / (hi - lo + 1e-8) * (zhi - zlo) + zlo, y)
    return loss_fn(name)(z, y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=list(VLM))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--losses", nargs="*", default=["ce"])
    ap.add_argument("--ctx-init", default=None)
    ap.add_argument("--lp", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t, classes = load_table(a.bench)
    y = t.y.to_numpy()
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        done = {(r["setting"], r["model"], r["loss"], r["seed"]) for r in map(json.loads, res.read_text().splitlines())}
    arm = ("coop-lp" if a.lp else "coop") + ("-ctxinit" if a.ctx_init else "")
    with open(res, "a") as fh:
        for name in a.models:
            X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            model, _, tok = build_clip(name)
            model = model.to(dev).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            scale = model.logit_scale.exp().item()
            W0 = torch.tensor(text_weights(model.to("cpu"), tok, classes, "clip_ensemble"), device=dev)
            model = model.to(dev)
            for lname in a.losses:
                if (f"{arm}-ensemble", name, lname, 0) in done:
                    continue
                all_z = []
                for seed in a.seeds:
                    torch.manual_seed(seed)
                    zf = out / f"logits_{arm}_{name}_{lname}_s{seed}.npy"
                    if (arm, name, lname, seed) in done and zf.exists():
                        all_z.append(np.load(zf))
                        continue
                    pr = Prompter(model, tok, classes, dev, a.ctx_init).to(dev)
                    idx = sample_shots(t, seed)
                    xb, yb = torch.tensor(X[idx], device=dev), torch.tensor(y[idx], device=dev)
                    zs = scale * xb @ W0.T
                    opt = torch.optim.SGD([pr.ctx], lr=0.002, momentum=0.9)
                    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
                    for _ in range(a.steps):
                        w = F.normalize(pr().float(), dim=-1)
                        loss = train_loss(lname, scale * xb @ w.T, yb, zs)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        sch.step()
                    with torch.no_grad():
                        w = F.normalize(pr().float(), dim=-1)
                    if a.lp:
                        H = torch.nn.Parameter(scale * w.clone())
                        o2 = torch.optim.AdamW([H], lr=1e-3, weight_decay=0.0)
                        for _ in range(a.steps):
                            loss = train_loss(lname, xb @ H.T, yb, zs)
                            o2.zero_grad()
                            loss.backward()
                            o2.step()
                        Z = X @ H.detach().cpu().numpy().T
                    else:
                        Z = scale * X @ w.cpu().numpy().T
                    all_z.append(Z)
                    np.save(zf, Z)
                    for r in evaluate(Z, t, {"setting": arm, "model": name, "loss": lname, "seed": seed}, n_boot=0):
                        fh.write(json.dumps(r) + "\n")
                    print(arm, name, lname, "seed", seed, f"loss {loss.item():.3f}", flush=True)
                for r in evaluate(np.mean(all_z, 0), t, {"setting": f"{arm}-ensemble", "model": name, "loss": lname, "seed": 0}, n_boot=0):
                    fh.write(json.dumps(r) + "\n")
                print(arm, name, lname, "ensemble", f"val bAcc {r['validation']['bacc']:.3f}", flush=True)
                fh.flush()


if __name__ == "__main__":
    main()
