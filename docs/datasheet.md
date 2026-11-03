# Datasheet for FundusShift v1.1

Follows Gebru et al., *Datasheets for Datasets*. "Not known" means the release cannot state it, not that the answer is no.

## Motivation

- **Purpose.** Measure how models for colour fundus screening hold up, in accuracy and calibration, on whole collections never seen in development.
- **Creators and funding.** To be filled in by the authors.

## Composition

- **Instances.** One colour fundus photograph per row. 35,280 released of 40,414 inventoried, from 15 collections. Each exclusion and its reason is in `metadata/splits_final.csv`.
- **Labels.** N (apparently normal), D (diabetic-retinopathy-like), G (glaucoma), plus an open-set split outside these classes. Labels describe the photograph as annotated by the source. They are not confirmed diagnoses.
- **Splits.** train 23,805, validation 2,983, calibration 2,968, test 3,320 (7 held-out collections), test_ood 1,343 (Chákṣu), test_openset 861.
- **Sample or complete.** A curated sample of openly licensed collections. It does not represent any population.
- **Demographics.** Age and sex appear only in PAPILA among the files we hold. Fairness by subgroup cannot be assessed across the benchmark.
- **Sensitive data.** Medical images released by their sources. Some originals carry burned-in text (for example dates or eye tags), visible in samples. Whether any is identifying has not been audited.
- **Known problems.** See Limitations.

## Collection and preprocessing

- **Collection.** Taken from public releases. Devices, countries and licences are per collection in the card. Consent and ethics approval rest with each source. What each source reports is listed in `docs/ethics_by_source.md`; no statement was found or opened for OIA-DDR, IDRiD, HRF, Bangladesh Eye and AFIO.
- **Preprocessing.** Crop to the illuminated field, longest side at most 1024 px, JPEG q95. Original SHA-256 kept.
- **Duplicates.** Exact hash, then vessel-pattern NCC at threshold 0.70, joined with patient IDs into leakage groups that never span two splits.

## Uses

- **Intended.** Research on domain shift, calibration, selective prediction and open-set referral.
- **Not intended.** Clinical use or any deployment decision.
- **Risks.** Pooled accuracy hides failure by collection and class. Report per collection and class.

## Limitations

- Grading scales differ across sources. N means "no DR" in DR-only collections.
- Patient identifiers exist only for some collections.
- HYGD holds only G and PALM only N in test, so per-source scores there are one-class recall. HRF has 15 images per class.
- Within several collections, trivial image statistics predict the class (for example FIVES). Quality matching reduced this only partly and failed on small collections.
- Labels are kept as published by each collection; no label was changed. No independent clinical re-grading was done, so the rate of label error is unknown. A pilot with AI raters was too weak on glaucoma to estimate it (kappa with the published labels 0.30 to 0.38).
- Pretrained models may have seen held-out collections: FLAIR's list includes FIVES, HRF, PAPILA, Paraguay and Chákṣu.
- Confident learning flags are candidates only. The detector found about 57 % of synthetic flips at about 26 % precision.

## Distribution and licences

- Hugging Face, private at present: `le-dang1haha/FundusShift`. No DOI yet.
- No single licence. Each image keeps its source licence. OIA-DDR (CC BY-NC-SA 4.0, 12,520 images), DeepDRiD (CC BY-SA 4.0, 2,000) and PAPILA (GPL-3.0-or-later, 420) carry the strongest terms. Published terms allow research use and non-commercial redistribution with attribution, stated changes and the same licence. Use of the full benchmark is non-commercial.
- Not legal advice. PAPILA's licence was confirmed from a University of Murcia record because figshare could not be opened; check the figshare field before public release. Commercial use of models trained on OIA-DDR is an open question for its authors.

## Maintenance

- Contact and errata process: to be filled in by the authors.
- Versioned releases with checksums; changes to splits get a new version.
