# SIA

## Contributor
- **Name:** Aditi Raj
- **College:** IGDTUW
- **GitHub:** `aditiraj2006`

## Original paper / basis
- **Title:** Structure Invariant Transformation for better Adversarial Transferability
- **Authors:** Xiaosen Wang, Zeliang Zhang, Jianping Zhang
- **Venue:** ICCV 2023
- **Link:** https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Structure_Invariant_Transformation_for_better_Adversarial_Transferability_ICCV_2023_paper.pdf

## Implementation note
- CNN-side face-verification adaptation of the pure SIA transformation strategy.
- This folder is separate from `SIA_MI_TI`, which combines the SIA idea with MI-FGSM and TI-FGSM style updates.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `SIA`

## Slide
- Student presentation is provided in `slides/student_presentation.pdf` when available.
