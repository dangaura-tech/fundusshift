# External cohorts: protocol

Written on 5 November 2026, before any external image was scored. Code: `experiments/external.py`.

## Cohorts
| Cohort | Country | Images | Labels used | Class map |
|---|---|---|---|---|
| Messidor-2 | France | 1,744 gradable (adjudicated ICDR grades, public Hugging Face copy `sngsfydy/Messidor2`) | ICDR 0-4 | 0 -> N, 1-4 -> D (as the benchmark maps ICDR) |
| G1020 | Germany | 1,020 (724 normal, 296 glaucoma) | glaucoma yes/no | 0 -> N, 1 -> G |

They are reported as **two separate checks**, not one pooled 3-class cohort, because class and source are confounded within each.
Messidor-2 normals are DR-only normals (glaucoma not assessed); G1020 images have no DR grading.

## Frozen models (published split, nothing refit on external data)
- LOTUS (`no_projection`, seeds 0-2), initial design (`full`, seeds 0-2), MedSigLIP zero-shot (text part of LOTUS).
  Heads are not stored, so they are refit with the original seeds and data; the run is accepted only if the benchmark rows reproduce `results/revision/final/*_s{0,1,2}.npy`.
- ConvNeXt-T CE and ConvNeXt-T Group DRO, seeds 0-2, from the saved `best.pt`.
- Temperature from Cal-A; 80% coverage threshold and 90% LAC quantile from the same half of Cal-B as the paper; specificity thresholds (90%, 95%) on Cal-B.

## Preprocessing and overlap
- Same crop and resize as `fsbench.process_images` (black border removed, longest side 1024, JPEG q95), same MedSigLIP and ConvNeXt transforms. Feature extraction is checked on 64 benchmark rows against the stored arrays.
- Near copies of benchmark images: nearest benchmark image by pooled MedSigLIP cosine; pairs above 0.95 are checked with pHash (distance <= 6 = duplicate) and dropped from scoring.
- Pretraining: MedSigLIP's fundus pretraining data is EyePACS (model card); neither cohort is listed (see `pretraining_overlap.md`).

## Reported, once
bAcc and per-class recall; rate of predictions in the class absent from the cohort (G on Messidor-2, D on G1020);
AUROC (disease vs normal; referable DR, ICDR >= 2, on Messidor-2); per-grade recall on Messidor-2; ECE and NLL after TS;
coverage and risk at the 80% threshold; LAC coverage at the 90% target; specificity and sensitivity at Cal-B thresholds for 90% and 95% specificity.
Paired bootstrap (2,000 resamples) of bAcc, LOTUS minus each comparator, seed-averaged probabilities.
Nothing is changed after the results are seen.
