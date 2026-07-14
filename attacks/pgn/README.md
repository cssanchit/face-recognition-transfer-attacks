# PGN

## Contributor
- **Name:** Chidroopa Kanaparthy
- **College:** Maharaja Agrasen Institute of Technology (MAIT)
- **GitHub:** `Chidroopakanaparthy`

## Original paper / basis
- **Title:** Boosting Adversarial Transferability by Achieving Flat Local Maxima
- **Authors:** Yao Ge, Yao Xiao, Weijie Huang, Xiaosen Wang, Yong Liu, Bo Han, Jiang Bian, Chunhua Shen
- **Venue:** NeurIPS 2023
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/fbfe5f1d5474d9f4e0992852a3c7e171-Abstract-Conference.html

## Implementation note
- CNN-side face-verification adaptation of PGN (Penalizing Gradient Norm).
- The adaptation keeps the neighbor-based two-gradient update idea while replacing the original classification objective with the shared embedding-space verification loss used in this repository.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `PGN`

## Slide
- Student presentation is provided in `slides/student_presentation.pdf` when available.
