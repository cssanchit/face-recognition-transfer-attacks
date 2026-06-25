# DPA_HMA_ENSEMBLE

## Contributor
- **Name:** Shared repository extension
- **College:** N/A
- **GitHub:** `AItruste`

## Original paper / basis
- **Title:** Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation
- **Authors:** Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang
- **Venue:** CVPR 2025
- **Link:** https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Improving_the_Transferability_of_Adversarial_Attacks_on_Face_Recognition_with_CVPR_2025_paper.html

## Implementation note
- This repository includes an ensemble extension built on top of the shared `DPA_HMA` adaptation. It combines multiple surrogate models inside the same CNN face-verification pipeline and should be read as a repository-side extension rather than a separate original paper method.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `DPA_HMA_ENSEMBLE`

## Slide
- No separate student presentation is attached for this repository-side extension.
