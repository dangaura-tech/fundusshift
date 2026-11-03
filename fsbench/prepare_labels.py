"""Step 0: convert each open source's native annotation files into <raw>/<source>/labels.csv

Columns: image (path relative to the source dir), label ('|'-separated native tokens), patient, eye, grade (ICDR 0-4)
Native tokens are written verbatim (prefixed with the column / scale they come from) so nothing is lost.
python -m fsbench.prepare_labels --raw data/raw [--only hygd grape ...]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _imgs(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG and not p.name.startswith("._"))


def _rows_to_df(rows):
    return pd.DataFrame(rows, columns=["image", "label", "patient", "eye", "grade"])


def hrf(src):
    tag = {"h": "healthy", "dr": "diabetic_retinopathy", "g": "glaucoma"}
    rows = []
    for p in _imgs(src / "images"):
        m = re.match(r"(\d+)_(h|dr|g)\.", p.name, re.I)
        rows.append([str(p.relative_to(src)), tag[m.group(2).lower()], "", "", ""])
    return _rows_to_df(rows)


def hygd(src):
    d = pd.read_csv(src / "Labels.csv").dropna(axis=1, how="all")
    return _rows_to_df([[f"Images/{r['Image Name']}", r["Label"], str(r["Patient"]), "", ""] for _, r in d.iterrows()])


def grape(src):
    rows = []
    for p in _imgs(src / "CFPs"):
        pid, eye, visit = p.stem.split("_")[:3]
        rows.append([str(p.relative_to(src)), "glaucoma_follow_up", pid, {"OD": "right", "OS": "left"}.get(eye, eye), ""])
    return _rows_to_df(rows)


def sustech_sysu(src):
    d = pd.read_csv(src / "originalImages/drLabels.csv")
    c5 = pd.read_csv(src / "originalImages/c5_DR_reclassified.csv").set_index("Fundus_images").iloc[:, 0].to_dict()
    rows = []
    for _, r in d.iterrows():
        g = int(r.iloc[2])
        icdr = c5.get(r.iloc[0], "") if g == 5 else g          # 5 = laser scars; reclassified ICDR grade if given
        tok = f"icdr_{g}" if g < 5 else f"laser_scars(reclassified_icdr_{icdr})"
        rows.append([f"originalImages/{r.iloc[0]}", tok, "", ["left", "right"][int(r.iloc[1])], str(icdr)])
    return _rows_to_df(rows)


def deepdrid(src):
    base = src / "DeepDRiD-master/regular_fundus_images"
    rows = []
    for part in ("training", "validation"):
        d = pd.read_csv(base / f"regular-fundus-{part}/regular-fundus-{part}.csv")
        for _, r in d.iterrows():
            g = r["left_eye_DR_Level"] if pd.notna(r["left_eye_DR_Level"]) else r["right_eye_DR_Level"]
            rel = f"DeepDRiD-master/regular_fundus_images/regular-fundus-{part}/Images/{r['patient_id']}/{r['image_id']}.jpg"
            q = "good_quality" if int(r["Overall quality"]) == 1 else "poor_quality"
            rows.append([rel, f"icdr_{int(g)}|{q}", str(r["patient_id"]), "left" if "_l" in r["image_id"] else "right", str(int(g))])
    t = pd.read_excel(base / "Online-Challenge1&2-Evaluation/Challenge1_labels.xlsx")
    for _, r in t.iterrows():
        pid = r["image_id"].split("_")[0]
        rel = f"DeepDRiD-master/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/{pid}/{r['image_id']}.jpg"
        rows.append([rel, f"icdr_{int(r['DR_Levels'])}", pid, "left" if "_l" in r["image_id"] else "right", str(int(r["DR_Levels"]))])
    return _rows_to_df(rows)


def rfmid2(src):
    rows = []
    for part, f in (("Training_set", "RFMiD_2_Training_labels.csv"), ("Validation_set", "RFMiD_2_Validation_labels.csv"),
                    ("Test_set", "RFMiD_2_Testing_labels.csv")):
        d = pd.read_csv(src / part / f, encoding="latin-1")
        d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
        cols = [c for c in d.columns if c != "ID"]
        for _, r in d.iterrows():
            toks = [c.strip() for c in cols if r[c] == 1]
            img = next((p for p in (src / part).glob(f"{r['ID']}.*") if p.suffix.lower() in IMG), None)
            rows.append([str(img.relative_to(src)) if img else f"{part}/{r['ID']}.jpg", "|".join(toks), "", "", ""])
    return _rows_to_df(rows)


def idrid(src):
    base = "B. Disease Grading"
    rows = []
    for gt, imgdir in (("a. IDRiD_Disease Grading_Training Labels.csv", "1. Original Images/a. Training Set"),
                       ("b. IDRiD_Disease Grading_Testing Labels.csv", "1. Original Images/b. Testing Set")):
        d = pd.read_csv(src / base / "2. Groundtruths" / gt).dropna(axis=1, how="all")
        d.columns = [c.strip() for c in d.columns]
        for _, r in d.iterrows():
            g = int(r["Retinopathy grade"])
            rows.append([f"{base}/{imgdir}/{r['Image name']}.jpg", f"icdr_{g}|dme_risk_{int(r['Risk of macular edema'])}",
                         "", "", str(g)])
    return _rows_to_df(rows)


def papila(src):
    base = next(src.glob("PapilaDB*"))
    diag = {0: "healthy", 1: "glaucoma", 2: "suspect"}
    rows = []
    for eye, f in (("right", "patient_data_od.xlsx"), ("left", "patient_data_os.xlsx")):
        d = pd.read_excel(base / "ClinicalData" / f, header=None).iloc[3:]
        for _, r in d.iterrows():
            pid = str(r[0]).lstrip("#")
            img = base / "FundusImages" / f"RET{pid}{'OD' if eye == 'right' else 'OS'}.jpg"
            rows.append([str(img.relative_to(src)), diag[int(r[3])], pid, eye, ""])
    return _rows_to_df(rows)


def jichi(src):
    d = pd.read_csv(src / "dmr/list.csv")
    rows = []
    for _, r in d.iterrows():
        pid, _, eye = Path(r["Image"]).stem.split("_")
        rows.append([f"dmr/{r['Image']}", f"davis_{r['Davis_grading_of_one_figure'].strip().lower()}", pid,
                     {"R": "right", "L": "left"}[eye], ""])
    return _rows_to_df(rows)


def fives(src):
    tag = {"A": "amd", "D": "diabetic_retinopathy", "G": "glaucoma", "N": "normal"}
    rows = []
    for p in _imgs(src):
        m = re.fullmatch(r"\d+_([ADGN])", p.stem)
        if "Original" in p.parts and m:
            rows.append([str(p.relative_to(src)), tag[m.group(1)], "", "", ""])
    return _rows_to_df(rows)


def _class_folders(src, base):
    """One class per sub-folder; token = folder name with spaces -> '_' (no patient information released)."""
    return _rows_to_df([[str(p.relative_to(src)), p.parent.name.replace(" ", "_"), "", "", ""]
                        for p in _imgs(src / base)])


def bangladesh_eye(src):
    return _class_folders(src, "Original Dataset")


def paraguay_dr(src):
    """7-level Paraguay scale -> ICDR: 1 none, 2 mild, 3 moderate, 4/5 severe & very severe NPDR, 6/7 PDR."""
    d = _class_folders(src, "")
    icdr = {"1": "0", "2": "1", "3": "2", "4": "3", "5": "3", "6": "4", "7": "4"}
    d["grade"] = [icdr.get(t.split(".")[0], "") for t in d.label]
    return d


def chaksu(src):
    """Majority decision of 5 experts per device; native token keeps the vote, e.g. glaucoma_suspect|votes_3of5."""
    rows = []
    for part in ("Train", "Test"):
        dec = src / part / "6.0_Glaucoma_Decision"
        for dev in ("Bosch", "Forus", "Remidio"):
            f = dec / f"Glaucoma_Decision_Comparison_{dev}_majority.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, encoding="utf-8-sig")
            imgs = {p.stem: p for p in _imgs(src / part / "1.0_Original_Fundus_Images" / dev)}
            experts = [c for c in d.columns if c.lower().startswith("expert")]
            for _, r in d.iterrows():
                stem = r["Images"].split(".")[0]
                if stem not in imgs:
                    continue
                norm = lambda v: re.sub(r"\s+", "_", str(v).strip().lower()).replace("susupect", "suspect")
                maj = norm(r["Glaucoma Decision"] if "Glaucoma Decision" in d.columns else r["Majority Decision"])
                votes = sum(norm(r[c]) == maj for c in experts)
                m = re.match(r"(P\d+)_", stem)
                rows.append([str(imgs[stem].relative_to(src)), f"{maj}|votes_{votes}of{len(experts)}|device_{dev}",
                             f"{dev}:{m.group(1)}" if m else "", "", ""])
    return _rows_to_df(rows)


def palm(src):
    """PALM: Label 1 = pathologic myopia. Only the training part names non-PM images as N (normal) or H (high myopia);
    validation/testing non-PM images may be either, so they are kept as 'non_pm' (excluded from N)."""
    rows = []
    for part in ("Training", "Validation", "Testing"):
        d = pd.read_excel(src / "PALM" / part / "Classification Labels.xlsx")
        sup = pd.read_excel(src / "PALM" / part / "Supplementary Information.xlsx").set_index("imgName").iloc[:, 0]
        for _, r in d.iterrows():
            name = r["imgName"]
            if int(r["Label"]) == 1:
                tok = "pathologic_myopia"
            else:
                tok = {"N": "normal", "H": "high_myopia_no_pm"}.get(name[0], "non_pm")
            dev = {1: "Zeiss", 2: "Canon"}.get(int(sup.get(name, 0)), "")
            rows.append([f"PALM/{part}/Images/{name}", f"{tok}|device_{dev}", "", "", ""])
    return _rows_to_df(rows)


def afio_composite(src):
    """AFIO (Rawalpindi) composite fundus+OCT set: colour fundus photographs only; class = folder, patient = MRN prefix."""
    base = src / "Dataset/Dataset"
    rows = []
    for p in _imgs(base):
        if "_Color_" not in p.name:
            continue
        group, cls = p.relative_to(base).parts[:2]
        eye = "right" if "_R_" in p.name else "left"
        rows.append([str(p.relative_to(src)), f"{group.lower()}_{cls.replace(' ', '_').lower()}", p.name.split("_")[0], eye, ""])
    return _rows_to_df(rows)


def ddr(src):
    """OIA-DDR DR grading split files: '<image> <grade>' (0-4 ICDR, 5 = ungradable)."""
    base = next(src.rglob("DR_grading"))
    rows = []
    for part in ("train", "valid", "test"):
        d = pd.read_csv(base / f"{part}.txt", sep=" ", header=None, names=["image", "grade"], dtype=str)
        for _, r in d.iterrows():
            g = r["grade"].strip()
            rows.append([str((base / part / r["image"]).relative_to(src)), f"ddr_{g}" if g != "5" else "ungradable",
                         "", "", g if g != "5" else ""])
    return _rows_to_df(rows)


PREPARERS = {"ddr": ddr, "palm": palm, "afio_composite": afio_composite, "chaksu": chaksu, "idrid": idrid, "papila": papila, "jichi": jichi, "fives": fives, "bangladesh_eye": bangladesh_eye,
             "paraguay_dr": paraguay_dr, "hrf": hrf, "hygd": hygd, "grape": grape, "sustech_sysu": sustech_sysu, "deepdrid": deepdrid,
             "rfmid2": rfmid2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    for sid, fn in PREPARERS.items():
        src = Path(a.raw) / sid
        if (a.only and sid not in a.only) or not src.is_dir():
            continue
        df = fn(src)
        df.to_csv(src / "labels.csv", index=False)
        miss = sum(not (src / p).is_file() for p in df.image)
        print(f"{sid:<14} rows={len(df):>6} missing={miss:>4} top labels={df.label.value_counts().head(6).to_dict()}")


if __name__ == "__main__":
    main()
