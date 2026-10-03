# TSD-Diffusion

This repository currently provides the core implementation.
Complete experiment configurations and reproduction instructions
will be released upon acceptance.

TSD-Diffusion studies timestep-aware configuration search and student distillation for few-step image generation. The workflow is:

**timestep search → schedule-conditioned architecture search → dynamic execution-policy search → student distillation → sampling and evaluation**.

## Release scope

This is a partial, source-only release for inspection of the core implementation, not a complete reproduction package. Experiment and model-construction configuration files are not included at this stage. Some retained entry points import or load these withheld files and are therefore not directly runnable as distributed. Complete configurations and stage-by-stage reproduction instructions will accompany the full release.

This repository does not contain experimental or evaluation results, searched configurations, generated images, logs, datasets, checkpoints, pretrained weights, or FID reference-statistics files. Evaluation functions and runtime output definitions are retained as source code; they are not saved experiment results. Some source modules contain default arguments, which should not be interpreted as the complete configurations used for the manuscript experiments.

## Directory guide

The upstream-compatible layouts are retained to preserve the source structure.

| Directory / file | Role in the workflow |
| --- | --- |
| `examples/guided_diffusion/search_imagenet64_classifier_guidance_optimized.py` | Stage I: timestep search for class-conditional ADM |
| `examples/guided_diffusion/guided_diffusion/nas/` | Stage II: schedule-conditioned architecture evaluation and dynamic execution-policy search |
| `examples/guided_diffusion/guided_diffusion/` | ADM model, policy-controlled U-Net, diffusion samplers, student distillation and evaluation |
| `examples/guided_diffusion/joint_search_fair_nsga2.py` | Joint-search comparison implementation |
| `examples/guided_diffusion/evaluations/` | TensorFlow Inception evaluation source for the ADM timestep-search path |
| `examples/Stable Diffusion/scripts/search_ea_new.py` | Stage I: timestep search for latent text-to-image generation |
| `examples/Stable Diffusion/ldm/` | Latent-diffusion U-Net, VAE, text encoder, sampler and data-loader source dependencies |
| `examples/Stable Diffusion/taming-transformers/` | Vector-quantization dependency imported by the autoencoder |
| `sd_autodiffusion_distill/distill/` | Teacher loading, configuration-guided student distillation, generation and evaluation source; experiment configuration modules are withheld |
| `sd_autodiffusion_distill/distill/nas/` | SD architecture space, dynamic policy representation, evolutionary selection and proxy evaluation |
| `sd_autodiffusion_distill/distill/adapters/` | Maps dynamic execution controls to the SD U-Net |
| `sd_autodiffusion_distill/distill/utils/` | Reads externally supplied COCO generation conditions |
| `scripts/export_timesteps.py` | Converts a locally produced Pareto front to a timestep array |
| `scripts/sd_pipeline.py` | Stage dispatch source; requires the withheld configuration modules to execute experiments |
| `src/pytorch_fid/` | PyTorch Inception feature extraction and FID utilities |
| `src_imagenet/` | Launcher source for the TensorFlow ImageNet evaluator |
| `evaluation_metrics.py` | FID utilities and paired image-text CLIP cosine similarity |
| `imagenet64_evaluation_metrics.py` | ImageNet64 metric utilities using externally supplied reference statistics |
| `tests/` | Static packaging and synthetic helper checks, not experiment results |
| `licenses/` | Upstream license texts; see `THIRD_PARTY_NOTICES.md` |

## Datasets and models

| Setting | Dataset | Model | Generation conditions |
| --- | --- | --- | --- |
| Pixel-space generation | ImageNet at 64 × 64 resolution | Class-conditional ADM U-Net with the corresponding classifier (ADM-G) | ImageNet class labels |
| Latent-space generation | COCO2014 at 512 × 512 resolution | Stable Diffusion v1.5 U-Net, VAE and text encoder | COCO captions |

Datasets and pretrained models must be obtained separately from their respective providers, subject to their licenses. No datasets, captions, weights or real-image reference statistics are distributed here. FID calculation requires reference statistics compatible with the feature extractor, preprocessing and real-image split used for evaluation; the TensorFlow and PyTorch extraction paths should not be assumed interchangeable.

## CLIP evaluation protocol

Text-to-image semantic alignment is measured using **CLIP ViT-L/14**. For each generated image and its corresponding generation prompt, image and text embeddings are L2-normalized, their cosine similarity is computed, and the similarities are averaged over the evaluated pairs. The metric is mean CLIP cosine similarity, without multiplication by 100 or positive-part CLIPScore scaling. Some legacy source identifiers use the name `CLIPScore`.

The CLIP model and tokenizer are external resources and are not included in this release.

## Verification scope

Dependency files are provided as source environment references, not as a tested environment lockfile. This release has undergone Python syntax, packaging and synthetic helper checks. These checks do not validate full GPU execution or reproduce manuscript results.

The configuration-independent checks can be run with:

```bash
python -B tests/check_release.py
python -B tests/test_helpers.py
python -B tests/test_search_utilities.py
```

Only load trusted pickle/checkpoint files. Review `THIRD_PARTY_NOTICES.md` and the upstream terms before redistribution.
