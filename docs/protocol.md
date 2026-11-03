# FundusShift: evaluation and selection protocol

Fixed on 31 October 2026, after interim test results had been seen. This is **not a pre-registration**. The paper must say so.

## Data use

| Split | Used for | Never used for |
|---|---|---|
| train | fitting | |
| validation | checkpoint, hyperparameters, prompt bank, choice of representative | calibration |
| calibration, Cal-A | temperature for TS and FTS | model choice |
| calibration, Cal-B | thresholds and conformal quantiles, separately for raw, TS and FTS | temperature |
| test, test_ood (Chákṣu), test_openset | reporting only | any choice or fit |

## Selection rule

**Primary rule for every setting that has a choice to make: highest validation balanced accuracy.**

| Setting | Primary | Secondary, reported as sensitivity |
|---|---|---|
| Fine-tuning, linear probe, 16-shot probe, MLP probe | best epoch by validation bAcc | ResNet-50 and ConvNeXt-T, CE, 3 seeds, by validation NLL (`runs/finetune_nll`) |
| Group DRO, CORAL, ERM | best epoch by validation bAcc | none |
| Tip-Adapter, Tip-Adapter-F | alpha, beta and bank by validation bAcc (`runs/tipadapter_bacc`) | official rule, validation accuracy (`runs/tipadapter`) |
| WiSE-FT | checkpoint and alpha by validation bAcc | validation NLL; alpha = 0.5; alpha = 1 |
| SWAD | method's own loss-valley rule on validation NLL. Stated exception: SWAD has no checkpoint to choose | none |
| CoOp and its ablations | no choice: last step of a fixed recipe | none |
| Zero-shot, MCQ | no checkpoint | none |

## Reporting

1. Every configuration is reported in appendix tables, raw, TS and FTS, with the same metrics.
2. A headline row per setting is its **representative**: the (model, config) with the highest validation bAcc, seed mean, raw scores (`experiments/paired_tests.py`). "Best" is never used for the highest test value.
3. Accuracy: pooled bAcc, equal-source bAcc, worst source. Calibration: ECE-15, MCE-15, ACE-15, NLL, Brier. Safety: D→N and G→N at frozen thresholds. Chákṣu is separate from test.
4. Comparisons: paired bootstrap over leakage groups, 2,000 draws, Holm within each metric family, on pairs fixed in `experiments/paired_tests.py`. Uncorrected p-values are not reported as findings.

## Pretraining overlap

FLAIR's published pretraining list includes FIVES, HRF, PAPILA, Paraguay and Chákṣu. FLAIR is reported apart and is not ranked with source-held-out models on full test. The fair comparison is the view `test_no_flair_overlap` (AFIO, HYGD, PALM; 838 images), which has few classes per source and must be read with that limit. Other models were checked by name in published documents only.

## Exploratory findings

These came from looking at test and must be labelled exploratory. They do not change the released splits or any recipe.

- Removing DDR from training raises glaucoma recall strongly on test. Class weighting does not explain it. Cause unknown.
- MedSigLIP test bAcc falls as training data grow (E7).

## Deviations

Record here any change to this protocol, with date and reason.

- 2 November 2026: proposed method added after baseline test results had been seen, and frozen in `docs/proposed_frozen_config.json` before its own test rows were read. About 120 configurations were compared by leave-one-development-source-out. Its paired tests are `experiments/proposed_paired.py`, against the representatives of `experiments/paired_tests.py`, and are not part of the original pair list. Any later variant run on test is post hoc.
- 3 November 2026: at the user's request, nine configurations with development bAcc of .791 or more were run on test and Chákṣu after the frozen method's test had been read (`experiments/top_dev_final.py`, `results/top_dev`). These are post hoc and do not replace the frozen method's numbers.
- 3 November 2026: LANTERN's fixed recipe was run on the 50 source rotations of `experiments/supplement.py` (`experiments/lantern_rotation.py`, `results/lantern_rotation.csv`). The rotations were not used to choose it.
- 3 November 2026, written before the rotation results of SCOPE were read: SCOPE (`experiments/scope.py`) treats a normal label from a DR-only collection as the label set {N, G}. It is adopted over LANTERN only if, on the 50 rotations with the same splits, (a) mean bAcc on sources that graded glaucoma ("full" scope) is higher on at least 30 of 50 rotations, (b) mean exact-label test bAcc is not lower by more than 0.010, and (c) mean ECE after TS is not higher by more than 0.010. Otherwise LANTERN stays the method and SCOPE is reported as an analysis. The published test is then read once for the adopted method.
- 3 November 2026, result: SCOPE failed rule (a): full-scope bAcc was higher on 10 of 50 rotations (mean -0.025); exact-label bAcc 0.761 against 0.771; ECE after TS 0.040 against 0.033. LANTERN stays. Rotation split 0 is the published split, so SCOPE's published test (bAcc 0.752, Chákṣu 0.609) was read as part of the rotation run; it is reported, not used for any choice.
- 3 November 2026, written before any result of `experiments/certify.py` was read: the LOSO-reference certificate replaces the Cal-B fit of temperature, thresholds and conformal quantiles only if, over the 50 rotations (split 0 excluded from the decision): (a) at the 80% target, |test coverage - 0.80| is smaller than for Cal-B on at least 30 of 50 rotations; (b) mean test risk at that threshold is not higher than Cal-B's; (c) mean LAC coverage on test is closer to 0.90 than Cal-B's. The normal certificate (p(N) threshold, 2% false reassurance on the reference) is reported as an add-on, judged by false reassurance (disease answered normal) on test at the cost of coverage.
- 3 November 2026, result of `experiments/certify.py` (split 0 excluded): (a) coverage at the 80% target closer to 0.80 on 32 of 50 rotations (mean 0.830 against Cal-B 0.744); (b) failed: mean test risk 0.141 against 0.117, since it answers more images; (c) LAC coverage 0.912 against 0.854. Rule not met, so Cal-B stays the protocol; the LOSO reference and the normal certificate are reported as analyses.
- 3 November 2026, second attempt, written before its results were read: the first rule compared risk at different coverages, which favours the arm that answers less. This attempt asks whether a promised risk holds. Threshold: SGR bound of 10% (delta 0.05), fitted on the LOSO reference or on Cal-B. The LOSO-reference certificate is adopted if, over the 50 rotations (split 0 excluded): (a) test risk is at most 0.10 on at least as many rotations as with Cal-B; (b) mean test coverage is higher than with Cal-B; (c) LAC coverage is closer to 0.90 (already met in the first attempt, kept as a condition). Risk at matched coverage is reported as a check that both arms rank images alike. This is the second rule tried for this component, which the paper must state.
- 3 November 2026, result of the second attempt (split 0 excluded): (a) met, test risk at most 0.10 on 44 of 50 rotations against 23 for Cal-B; (b) failed, mean test coverage 0.555 against 0.641; (c) met. Rule not met; Cal-B stays the protocol. Risk at matched coverage was identical for both arms (0.064, 0.106, 0.163 at 50/70/90%), confirming that the arms differ only in threshold placement. Both attempts are reported, with the LOSO reference described as keeping promises at a cost in coverage.
