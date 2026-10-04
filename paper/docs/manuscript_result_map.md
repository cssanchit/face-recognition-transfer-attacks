# Manuscript Section to Result Map

This note links the main manuscript sections, tables, and figures to the repository files that support them. The manuscript itself is maintained outside this repository.

## Main benchmark tables

- Attack-wise comparison table:
  - `results_summary/pairwise_split_summary.csv`
- Cross-model transferability table:
  - `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- RobFR external validation table:
  - `results_summary/robfr_external_validation_all_attacks.csv`
- Sibling-Attack validation table:
  - `results_summary/sibling_attack_validation_summary.csv`
- Ablation table:
  - `results_summary/ablation_overall_summary.csv`

## Main benchmark figures

- Surrogate-wise breach comparison:
  - derived by averaging valid attacker-victim rows from `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- Dataset-wise breach comparison:
  - `results_summary/datasetwise_summary.csv`
- Lambda sensitivity figure:
  - `results_summary/lambda_sweep_summary.csv`

## Notes

- The final surrogate-wise values are not backed by a separate `modelwise_summary.csv`. They are derived from the broader cross-model attacker-victim summary.
- The current attack-wise summary file also uses the broader cross-model victim pool rather than the narrower common-victim slice. It averages over all valid surrogate--victim transfer pairs, excluding self-transfer and the `Facenet512 -> Facenet` same-family case.
- For the cross-model analysis, the four attackers are ArcFace, Facenet512, GhostFaceNet, and VGG-Face. The candidate victim set is Facenet, Facenet512, ArcFace, GhostFaceNet, IR152, and VGG-Face, with self-transfer excluded and the `Facenet512 -> Facenet` same-family pair omitted from the reported 19-pair analysis.
- The repository intentionally focuses on code and compact result summaries rather than the manuscript source.
