# Third-party notices

The bundled model implementations contain third-party source code. Upstream
sources and license notices are listed below. Original source headers are retained.

| Component | Upstream source | Notice / license |
|---|---|---|
| BAGEL model implementation | [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) | Apache-2.0, except separately marked files |
| OmniGen-2 implementation | [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) | Apache-2.0, except separately marked components |
| DiT-derived position/timestep helpers in BAGEL `modeling_utils.py` | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | CC BY-NC 4.0 |
| TaylorSeer cache helpers | [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) | Upstream GPL-3.0 notice |
| Triton layer-normalization implementation | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause |

License texts are supplied in `licenses/`. File-specific notices take precedence
over a dependency's general license. Transformer, scheduler, and tokenizer
sources also retain their embedded upstream notices.

The bundled model code includes package-local imports and chat/adapter
integration modifications. These files are not presented as pristine upstream
releases. Backbone weights and datasets are not distributed here and have
separate terms. This package does not assign a new blanket license to code or
weights owned by third parties, nor select a new license for the first-party code.
