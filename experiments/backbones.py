"""Model registry for the FundusShift experiments.

VM  (12): ImageNet / self-supervised vision models, via timm (uniform preprocessing and head removal).
RETINA (2): RETFound fundus foundation models, gated weights stored locally under gated_models/.
VLM (10): image-text models via open_clip; their image towers also serve as frozen features.
`pretrain_data` feeds the paper's contamination table.
"""
import argparse
import os
from pathlib import Path

import torch

GATED = Path(os.environ.get("FS_GATED", Path(__file__).resolve().parents[2] / "gated_models"))
VM = {  # name -> (timm id, pretraining data, fine-tuned in the benchmark?)
    "deit_b16": ("deit_base_patch16_224.fb_in1k", "ImageNet-1k", True),
    "convnext_t": ("convnext_tiny.fb_in1k", "ImageNet-1k", True),
    "swin_t": ("swin_tiny_patch4_window7_224.ms_in1k", "ImageNet-1k", True),
    "inception_v3": ("inception_v3.tv_in1k", "ImageNet-1k", True),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k", "ImageNet-1k", True),
    "densenet121": ("densenet121.tv_in1k", "ImageNet-1k", True),
    "vit_b16_in21k": ("vit_base_patch16_224.augreg_in21k", "ImageNet-21k", True),
    "resnet50": ("resnet50.tv2_in1k", "ImageNet-1k", True),
    "mobilenetv3_l": ("mobilenetv3_large_100.ra_in1k", "ImageNet-1k", True),
    "vgg16": ("vgg16.tv_in1k", "ImageNet-1k", True),
    "vit_b16_mae": ("vit_base_patch16_224.mae", "ImageNet-1k (MAE, self-supervised)", True),
    "dinov2_b14": ("vit_base_patch14_dinov2.lvd142m", "LVD-142M (self-supervised)", False),
    "retfound_mae": ("vit_large_patch16_224", "MEH-MIDAS + public fundus (MAE, RETFound)", True),
    "retfound_dinov2": ("vit_large_patch14_dinov2", "MEH-MIDAS fundus (DINOv2, RETFound)", True),
}
VLM = {  # name -> (open_clip arch, pretrained tag or None for hf-hub, pretraining data)
    "clip_b16": ("ViT-B-16", "openai", "OpenAI web image-text"),
    "metaclip_b16": ("ViT-B-16", "metaclip_400m", "MetaCLIP (web image-text)"),
    "siglip_b16": ("hf-hub:timm/ViT-B-16-SigLIP", None, "WebLI"),
    "siglip_l16": ("hf-hub:timm/ViT-L-16-SigLIP-256", None, "WebLI"),
    "siglip2_b32": ("hf-hub:timm/ViT-B-32-SigLIP2-256", None, "WebLI"),
    "siglip2_b16": ("hf-hub:timm/ViT-B-16-SigLIP2", None, "WebLI"),
    "siglip2_l16": ("hf-hub:timm/ViT-L-16-SigLIP2-256", None, "WebLI"),
    "bioclip_b16": ("hf-hub:imageomics/bioclip", None, "iNat/biodiversity (TreeOfLife-10M)"),
    "bioclip2_l14": ("hf-hub:imageomics/bioclip-2", None, "biodiversity (TreeOfLife-200M)"),
    "biomedclip": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None, "PMC-15M biomedical figures"),
}


def build_retfound(name, num_classes=0):
    """RETFound as in the official code: 224 px, ImageNet normalisation, pretrained encoder loaded by matching keys."""
    import timm
    from timm.layers import resample_abs_pos_embed
    kw = {"global_pool": "avg"} if name == "retfound_mae" else {}
    m = timm.create_model(VM[name][0], pretrained=False, img_size=224, num_classes=num_classes, **kw)
    torch.serialization.add_safe_globals([argparse.Namespace])  # RETFound MAE stores its training args
    ck = torch.load(GATED / name / f"{'RETFound_mae_natureCFP' if name == 'retfound_mae' else 'RETFound_dinov2_meh'}.pth",
                    map_location="cpu", weights_only=True)
    sd = ck.get("model", ck.get("teacher", ck))
    sd = {k.replace("backbone.", "").replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2."): v for k, v in sd.items()}
    own = m.state_dict()
    if "pos_embed" in sd and sd["pos_embed"].shape != own["pos_embed"].shape:
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], [16, 16], num_prefix_tokens=1)
    sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in sd and not k.startswith(("head", "fc_norm"))]
    if missing:
        raise RuntimeError(f"{name}: checkpoint does not cover {missing[:8]}")
    m.load_state_dict(sd, strict=False)
    cfg = dict(timm.data.resolve_model_data_config(m), input_size=(3, 224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return m, cfg


def build_vm(name, num_classes=0):
    import timm
    if name.startswith("retfound"):
        return build_retfound(name, num_classes)
    # same recipe for all: no stochastic depth (timm's Swin default is 0.1)
    m = timm.create_model(VM[name][0], pretrained=True, num_classes=num_classes, **({"drop_path_rate": 0.0} if name == "swin_t" else {}))
    cfg = timm.data.resolve_model_data_config(m)
    if cfg["input_size"][-1] < 224:     # tv2 ResNet-50 is trained at 176 but evaluated at 224
        cfg["input_size"] = (3, 224, 224)
    return m, cfg


def build_clip(name):
    import open_clip
    arch, tag, _ = VLM[name]
    if tag:
        model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag)
    else:
        model, _, pre = open_clip.create_model_and_transforms(arch)
    return model, pre, open_clip.get_tokenizer(arch)


def feature_extractor(name):
    """(model, preprocess, embed_fn) for frozen features."""
    if name in VM:
        import timm
        m, cfg = build_vm(name)
        return m, timm.data.create_transform(**cfg, is_training=False), lambda m, x: m(x)
    m, pre, _ = build_clip(name)
    return m, pre, lambda m, x: m.encode_image(x)


ALL = [*VM, *VLM]
