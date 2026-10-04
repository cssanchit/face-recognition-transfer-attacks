# RobFR + FaceSM Integration Notes

## Decision
Use RobFR as an external validation benchmark, not as a replacement for the current benchmark.

Reason:
- RobFR natively supports a different dataset/model stack than the current project.
- It is still a strong external baseline framework for demonstrating portability of FaceSM.

## Native RobFR support confirmed

### Black-box attacks
RobFR already has black-box entry points for:
- FGSM
- BIM
- MIM
- CIM

Relevant files:
- `robfr/RobFR/benchmark/FGSM_black.py`
- `robfr/RobFR/benchmark/BIM_black.py`
- `robfr/RobFR/benchmark/MIM_black.py`
- `robfr/RobFR/benchmark/CIM_black.py`
- `robfr/run_black.sh`

### Shared attack loss location
All four attacks inherit from `ConstrainedMethod`, so the cleanest SM integration point is:
- `robfr/RobFR/attack/base.py`

Current vanilla loss:
- `ConstrainedMethod.getLoss(features, ys)`
- impersonate: mean squared distance to target embedding
- dodging: negative mean squared distance to target embedding

### Model interface
RobFR face models share a common normalized embedding interface:
- `robfr/RobFR/networks/FaceModel.py`
- `forward()` returns normalized embeddings

This is a good place to add optional mirror-fused embedding support if we want to implement the full FaceSM variant.

### Dataset support
RobFR natively supports:
- LFW
- YTF
- CFP-FP

Relevant files:
- `robfr/RobFR/dataset/lfw.py`
- `robfr/RobFR/dataset/ytf.py`
- `robfr/RobFR/dataset/cfp.py`

## Important mismatch with current paper benchmark
Current project uses:
- datasets: LFW, CelebA, VGGFace2
- models: Facenet512, ArcFace, GhostFaceNet, VGG-Face, IR152

RobFR natively supports a different stack.

What appears usable without heavy porting:
- ArcFace
- FaceNet variants
- IR variants
- MobileFace / Mobilenet family

What does not look native:
- Facenet512
- GhostFaceNet
- VGG-Face
- CelebA pair protocol
- VGGFace2 pair protocol

## Recommended experiment path

### Main paper benchmark
Keep the current benchmark as the primary evaluation.

### External validation benchmark
Add a RobFR section with four black-box attacks:
- FGSM
- BIM
- MIM
- CIM

Recommended first setup:
- datasets: LFW and CFP-FP
- surrogate models: ArcFace and FaceNet-VGGFace2 or IR50-ArcFace
- victim models: MobileFace, Mobilenet, IR50 variant not used as surrogate

This gives a credible external check without forcing a full port of the current custom pipeline.

## Suggested FaceSM integration levels

### Option A: SS only
Minimal and fastest.
- Add source embedding to the loader output.
- Extend loss to use target and source embeddings.
- Keep model forward path unchanged.

Needed files:
- `robfr/RobFR/dataset/base.py`
- `robfr/RobFR/attack/base.py`
- black-box benchmark scripts to pass any new flags

### Option B: Full FaceSM (recommended)
Add both:
- source separation
- mirror-fused embedding

Likely implementation plan:
1. Add optional `forward_mirror()` or a `mirror_fuse` flag in `FaceModel.forward()`.
2. Loader returns both source and target embeddings.
3. `ConstrainedMethod.getLoss()` accepts current adv embedding, target embedding, source embedding.
4. Add `_SM` attack variants or a `--facesm` switch for FGSM/BIM/MIM/CIM.

## Why this is a good novelty test
If FaceSM improves RobFR attacks too, the paper can claim:
- FaceSM is not tied to the custom benchmark pipeline.
- FaceSM improves attacks inside an established external face-robustness framework.
- The method operates as an objective-level enhancement rather than a bespoke attack implementation.

## Best next coding step
Implement a small RobFR fork locally with:
- FGSM_SM
- BIM_SM
- MIM_SM
- CIM_SM

Start with:
- ArcFace surrogate
- LFW dataset
- one or two victim models

This is the lowest-risk proof-of-concept before scaling.
