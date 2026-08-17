# DPA_VMI

## Contributor
- **Name:** Amogh Saxena
- **College:** ADGIPS, Delhi
- **GitHub:** https://github.com/SaxenaAmogh

## Original papers / basis
- **DPA title:** Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation
- **DPA authors:** Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang
- **DPA venue:** CVPR 2025
- **DPA link:** https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Improving_the_Transferability_of_Adversarial_Attacks_on_Face_Recognition_with_CVPR_2025_paper.pdf

- **VMI title:** Enhancing the Transferability of Adversarial Attacks through Variance Tuning
- **VMI authors:** Xiaosen Wang, Kun He
- **VMI venue:** CVPR 2021
- **VMI link:** https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html

## Implementation note
`DPA_VMI` is a student-contributed hybrid adaptation for CNN-based face verification. It combines DPA-style projective/perspective transformed batches with VMI-style neighbor gradient variance estimation, Nesterov-style lookahead, and Gaussian gradient smoothing.

This implementation is an adaptation for the shared face-verification attack pipeline, not an official release from the original paper authors.

## Verification note
A partial verification was run on 6 balanced source-target pairs using `Facenet512`, `ArcFace`, and `GhostFaceNet` as attackers, giving 72 attacker-victim evaluation cases. On that partial check, `DPA_VMI` achieved `43.06%` breach rate and `0.2454` mean impact. The full 480-case result was not verified due to runtime cost, especially for the `VGG-Face` attacker.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `DPA_VMI`

## Student presentation
- [`student_presentation.pdf`](student_presentation.pdf)
