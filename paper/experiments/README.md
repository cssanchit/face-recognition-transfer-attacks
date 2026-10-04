# Experiment Scripts

This folder contains the main scripts and notes used to rebuild paper summaries and supporting analyses.

## Files

- `build_paper_results_lambda20_limit1000.py`: rebuilds the main paper-ready summary CSV files.
- `ablation_sm_experiment.py`: runs or summarizes the ablation study for FaceSM components.
- `lambda_sweep_sm_experiment.py`: runs or summarizes the source-separation weight sensitivity study.
- `finalize_facenet512_lambda20.py`: helper script used during the Facenet512 lambda-0.20 result finalization.
- `run_robfr_facesm_pilot.py`: runs the RobFR pilot evaluation for FaceSM.
- `run_robfr_lgc_extension.py`: extends the RobFR validation to include the LGC attack.
- `run_sibling_facesm_experiment.py`: prepares local image-pair paths and runs the Sibling-Attack and FaceSM-Sibling workflow.
- `robfr_sm_integration_notes.md`: notes on how FaceSM was integrated into the RobFR workflow.
- `emb_trajectory_grad_alignment_experiment.ipynb`: notebook used for embedding-trajectory and gradient-alignment analysis.

## Core dependencies

The scripts in this folder mainly rely on the `core/facesm_attack_core.py`, `core/evaluate_attack_performance.py`, `core/ir152_model.py`, and `core/verification_thresholds.json` files.
