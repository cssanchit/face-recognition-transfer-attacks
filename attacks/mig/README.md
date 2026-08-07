# MIG

## Contributor
- **Name:** Lakshita Sharma
- **College:** Bhagwan Parshuram Institute of Technology
- **GitHub:** [`slakshita04`](https://github.com/slakshita04)

## Original paper / basis
- **Title:** Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients
- **Authors:** Wenshuo Ma, Yidong Li, Xiaofeng Jia, Wei Xu
- **Venue:** ICCV 2023
- **Link:** https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html
- **Reference code:** https://github.com/Trustworthy-AI-Group/TransferAttack

## Implementation note
This repository includes a CNN face-verification adaptation of MIG. The classification objective from the original setting is replaced with the shared embedding-space cosine objective used by this codebase, while the attack keeps the integrated-gradient path idea and momentum update style.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `MIG`
