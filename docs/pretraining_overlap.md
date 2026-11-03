# Pretraining overlap with the FundusShift test sources

Test sources: AFIO, FIVES, HRF, HYGD, PALM, PAPILA, Paraguay; Chákṣu as the OOD view. Checked 2026-11-01. A listed name is evidence of overlap; an unlisted name is not proof of none.

| Model | Stated fundus pretraining data | Test sources named | Status |
|---|---|---|---|
| FLAIR | Own README list | FIVES, HRF, PAPILA, Paraguay, Chákṣu | Overlap. Report apart; use `test_no_flair_overlap` |
| MedSigLIP | Model card: EyePACS as its fundus set | None | No direct overlap stated |
| RETFound MAE (`natureCFP`) | Nature 2023: MEH-MIDAS 815,468 + Kaggle EyePACS 88,702 | None. PAPILA and IDRiD appear only in that paper's downstream evaluation | No direct overlap stated |
| RETFound DINOv2 (`meh`) | Repository links a preprint not read here; a 2025 arXiv preprint describes 904K AlzEye images, which differs in name from the Nature description | None found | Not verified from the primary source |
| Fundus-R1 | arXiv 2604.08322, Table 2. CFP: FGADR, IDRiD, GRAPE, JSIEC, Retinal-Lesions, RFMiD, OIA-ODIR, DDR, DeepDRiD; plus MMC-AMD | None. IDRiD, GRAPE, DDR, DeepDRiD are development sources here | No direct overlap with test stated; JSIEC and ODIR contents not traced |
| LLaVA-Med, BioMedCLIP | Biomedical literature figures | Unknown | Cannot be ruled out: figures from papers may include these collections |
| Qwen2.5-VL, LLaVA-OneVision, CLIP, SigLIP, DINOv2, MetaCLIP, BioCLIP | Web-scale or natural-image data | Unknown | Cannot be ruled out, not fundus-specific |

Sources: [RETFound, Nature 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550819/), [RETFound repository](https://github.com/rmaphoh/RETFound), [RETFound-DINOv2 preprint](https://arxiv.org/abs/2509.03421v1), [Fundus-R1](https://arxiv.org/pdf/2604.08322).
