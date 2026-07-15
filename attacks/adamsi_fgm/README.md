# ADAMSI_FGM

## Contributor
- **Name:** Bhumika Singh
- **College:** SRM Institute of Science and Technology

## Original paper / basis
- **Title:** On the Convergence of an Adaptive Momentum Method for Adversarial Attacks
- **Authors:** Yuxiao Long, Xuanyu Yi, Jinglun Xu, Yuan Wu, Xiaosen Wang
- **Venue:** AAAI 2024
- **Link:** https://ojs.aaai.org/index.php/AAAI/article/view/27997

## Implementation note
- CNN-side face-verification adaptation of ADAMSI_FGM.
- The adaptation modifies the standard MI-FGSM style update using an adaptive momentum coefficient and an adaptive per-pixel step based on accumulated squared gradients.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `ADAMSI_FGM`

## Slide
- Student presentation is provided in `slides/student_presentation.pdf` when available.
