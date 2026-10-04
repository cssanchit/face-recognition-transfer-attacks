# Result Summary Files

This folder contains compact CSV files used for manuscript verification.

## Files

- `pairwise_split_summary.csv`: attack-wise comparison for impersonation and dodging under the broader cross-model victim pool. It averages over all valid surrogate--victim transfer pairs, excluding self-transfer and the `Facenet512 -> Facenet` same-family case.
- `datasetwise_summary.csv`: dataset-wise averages on the main benchmark common victim pool.
- `cross_model_transferability_analysis_with_attacker_victim_pairs.csv`: broader attacker-victim transfer results with explicit pair scope. The reported analysis uses four attackers (`ArcFace`, `Facenet512`, `GhostFaceNet`, `VGG-Face`) and six candidate victims (`Facenet`, `Facenet512`, `ArcFace`, `GhostFaceNet`, `IR152`, `VGG-Face`), with self-transfer excluded and the `Facenet512 -> Facenet` same-family pair omitted from the final 19-pair summary. This file also serves as the numerical source for the final surrogate-wise Figure 2(a) values after averaging over valid unseen victims for each surrogate.
- `ablation_overall_summary.csv`: ablation summary for FaceSM components.
- `lambda_sweep_summary.csv`: sensitivity results over source-separation weight values.
- `robfr_external_validation_all_attacks.csv`: RobFR validation summary including BIM, MIM, CIM, and LGC.
- `sibling_attack_validation_summary.csv`: Sibling-Attack impersonation validation summary averaged over VGG-Face, Facenet512, and ArcFace victims.
