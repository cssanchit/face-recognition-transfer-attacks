# FaceSM Anonymous Review Repository

This repository is the anonymous review package for the FaceSM study on transferable adversarial attacks against face verification systems. It is organized to make the results and code easy to audit: the main reported numbers are backed by compact CSV summaries, and the core experiment scripts are included.

## Start Here

If you are reviewing the paper and want the quickest path through the materials:

1. Check `results_summary/` for the verified CSV files behind the main tables and figures.
2. Use `docs/manuscript_result_map.md` to trace manuscript sections to repository files.
3. Inspect `core/` for the main implementation.
4. Inspect `experiments/` for the scripts used to rebuild the summarized outputs.

## Repository Layout

- `results_summary/`
  - Compact CSV files backing the paper tables and figure values.
- `core/`
  - Main attack and evaluation code, plus utilities used during result consolidation.
- `experiments/`
  - Scripts and notebooks used to rebuild summary tables, ablations, sensitivity studies, and RobFR validation.
- `robfr_patch/`
  - Local RobFR-side modifications used for the external validation experiments.
- `Sibling-Attack/`
  - Sibling-Attack and FaceSM-Sibling workflow code. The large pretrained
    weights and generated adversarial images are intentionally excluded.
- `docs/`
  - Data-availability notes and a manuscript-to-results map.

## What Is Included

- verified result summary CSV files
- core implementation and experiment scripts

## What Is Not Included

To keep the review package lightweight and safe to share, the repository intentionally excludes:

- raw face dataset images
- large pretrained model weights
- full generated adversarial image dumps
- Sibling-Attack pretrained `.pth` weights
- the larger private workspace used during experiment execution

## Main Result Files

The primary manuscript numbers are backed by the following summary files:

- `results_summary/pairwise_split_summary.csv`
- `results_summary/datasetwise_summary.csv`
- `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- `results_summary/ablation_overall_summary.csv`
- `results_summary/lambda_sweep_summary.csv`
- `results_summary/robfr_external_validation_all_attacks.csv`
- `results_summary/sibling_attack_validation_summary.csv`

For the final manuscript, the surrogate-wise values shown in Figure 2(a) are derived by averaging the valid attacker-victim rows in `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`. A separate `modelwise_summary.csv` is therefore not part of this review package.

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested Reviewer Entry Points

- verified summaries: `results_summary/`
- section-to-results map: `docs/manuscript_result_map.md`
- summary rebuild script: `experiments/build_paper_results_lambda20_limit1000.py`
- RobFR extension script: `experiments/run_robfr_lgc_extension.py`
- Sibling workflow runner: `experiments/run_sibling_facesm_experiment.py`
