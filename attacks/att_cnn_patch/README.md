# ATT_CNN_PATCH

## Contributor
- **Name:** Pratyush Kumar
- **College:** KCC Institute of Technology and Management, A.K.T.U.

## Original paper / basis
- **Title:** Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning
- **Authors:** Di Ming, Peng Ren, Yunlong Wang, Xin Feng
- **Venue:** NeurIPS 2024
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html

## Implementation note
- CNN-side adaptation inspired by ATT.
- This is not an official reproduction of the original ViT token-level ATT implementation.
- This variant is separate from `ATT_CNN` and uses gradient-variance-based modulation with stochastic patch masking.

## Code location
- Shared implementation lives in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py)
- Attack registry name: `ATT_CNN_PATCH`

## Verified subset result
- Overall breach rate: `23.54%`
- Mean impact: `0.1476`
