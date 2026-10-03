# Third-party source and notices

This release retains modified upstream code. The root LICENSE is the original project's inherited MIT notice; it does not replace licenses of third-party files or pretrained models.

| Source subtree | Upstream | License text |
| --- | --- | --- |
| `examples/guided_diffusion/` | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | `licenses/OpenAI-guided-diffusion.txt` |
| `examples/Stable Diffusion/ldm/` | [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion) and [stable-diffusion](https://github.com/CompVis/stable-diffusion) | `licenses/CompVis-latent-diffusion.txt`, `licenses/CompVis-stable-diffusion.txt` |
| `examples/Stable Diffusion/taming-transformers/` | [CompVis taming-transformers](https://github.com/CompVis/taming-transformers) | `licenses/CompVis-taming-transformers.txt` |
| `src/pytorch_fid/` | [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) | `licenses/pytorch-fid.txt` |
| UniPC solver source | [wl-zhao/UniPC](https://github.com/wl-zhao/UniPC) | `licenses/UniPC.txt` |
| DPM-Solver solver source | [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver) | `licenses/DPM-Solver.txt` |

Local modifications include architecture and execution-policy controls, search and distillation integration, evaluator integration, and portable external-resource paths. Existing source attribution is retained. See file headers for more specific origins. No pretrained weights or datasets are distributed. Their terms must be checked separately by the user.
