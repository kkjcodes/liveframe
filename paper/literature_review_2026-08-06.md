# LiveFrame Literature Review and Publication Assessment

**Review cutoff:** 2026-08-06
**Method:** primary-source web review of arXiv records, proceedings records, official repositories, official documentation, and model cards. Non-discovery is not proof of absolute priority.

## Executive verdict

Publishing is useful, but the defensible contribution is narrower than an unqualified “first.” The strongest paper is an evidence-first systems and reproducibility study with one exact-output optimization and three bounded negative results.

- **Strongest novelty:** the combination and scale of the CUDA/MLX autoregressive video divergence protocol: byte-identical serialized stochastic inputs, 128 paired videos, two long horizons, and separate intrinsic-quality and same-seed identity decisions.
- **Most immediately useful engineering result:** the Apple M5 Max component profile and 1.2108x complete-wall speedup from a bounded MLX free-buffer-cache change with repeated exact pre-encode RGB digest records and zero swap.
- **Useful negative mechanism result:** the prospective rejection of one exact periodic raw-K/V anchor-substitution intervention. Broad causal-cache reuse and RoPE-aware memory are not new.
- **Expected impact:** methodological, reproducibility, and ecosystem utility rather than a state-of-the-art generation result. The paper can become a useful citation for claim discipline and cross-runtime video reproducibility if the evidence tiers are publicly accessible.

## Corrections to the initial sweep

1. Flash-VAED did not first appear on 2026-08-05. arXiv:2602.19161 v1 was submitted on 2026-02-22; v2 was posted on 2026-08-05.
2. The M5 baseline VAE share is **78.69%**. The optimized 4 GiB row is approximately **72.33%**. Describing both 72% and 78.69% as baseline values is incorrect.
3. “Give Me FP32” is not the current official title of arXiv:2506.09501. The verified title is *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*.
4. arXiv:2502.08202 is unrelated privacy work. The verified Rolling Forcing paper is arXiv:2509.25161.
5. Claims such as “no one has published Apple-Silicon video benchmarks” are too broad. Public Apple results exist, but the reviewed workloads are not apples-to-apples with LiveFrame. The safe claim is that no reviewed primary source matched the complete Apple/MLX/Wan/one-step/exact-output allocator-policy configuration.
6. The denoise-side Delta Scheduling oracle did not fail; it was never run. The 70.3667% figure is a frozen pre-measurement reclaim requirement.

## Cross-runtime numerical divergence

### Closest sources

| Source | Status | Relevant overlap | Important distinction |
| --- | --- | --- | --- |
| Yuan et al., *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*, arXiv:2506.09501 (2025) | arXiv; NeurIPS 2025 reported by the authors | Quantifies generative divergence from GPU type, count, batch size, and precision | Text-only; no Apple MLX/Metal, diffusion, or serialized stochastic payload |
| Pham et al., *CRADLE*, ICSE 2019, DOI 10.1109/ICSE.2019.00107 | Peer reviewed | Cross-backend differential testing and inconsistency localization | Detects framework bugs rather than long-horizon generative identity |
| Zhang et al., *The Emergence of Reproducibility and Generalizability in Diffusion Models*, ICML 2024, arXiv:2310.05264 | Peer reviewed | Same-noise deterministic diffusion comparison | Varies trained models, not hardware/backend execution of one frozen model |
| Cankaya, *Bit-Exact AI Inference Verification Without Performance Tradeoffs*, arXiv:2606.00279 (2026) | Workshop/preprint | Cross-accelerator bit-exact LLM reconstruction and execution fingerprints | NVIDIA and text generation only |
| Loose et al., *FloatDoor*, arXiv:2606.19535 (2026) | Preprint | Cross-platform numerical signatures affecting generation | Deliberately amplified in a compromised LLM |
| Möller et al., *Adversarial Inputs for Linear Algebra Backends*, DOI 10.60882/cispa.32771232 (2025) | Repository/conference contribution; archival status unclear | Intel MKL, NVIDIA CUDA, and Apple Accelerate discrepancies | Adversarial classifiers, not natural MLX video rollouts |

### Paper-safe novelty statement

> In the primary literature located through 2026-08-06, prior work establishes hardware-, software-, topology-, backend-, and precision-induced numerical divergence in neural-network inference; downstream divergence in generative LLMs; and same-noise consistency across separately trained diffusion models. We did not locate a controlled study matching LiveFrame’s combination of byte-identical serialized stochastic inputs, CUDA-versus-MLX/Metal execution of a frozen visual generator, 128 paired autoregressive video-diffusion rollouts at two temporal horizons, and separate numerical, intrinsic-quality, temporal, and perceptual-identity verdicts.

Do not claim the first proof that hardware affects generative inference.

## Apple Silicon and video-decoder characterization

### Closest sources

| Source | Date | Relevant result | Distinction |
| --- | --- | --- | --- |
| Zhu et al., *Flash-VAED*, arXiv:2602.19161 v1/v2 | 2026-02-22 / 2026-08-05 | Approximately 6x decoder acceleration and up to 36% end-to-end acceleration; decoder shares reported up to about 40.9% in accelerated NVIDIA pipelines | Distilled/pruned output-changing decoder, not Apple/MLX or exact-output allocator policy |
| `madebyollin/taehv`, TAEW2.1 | Public by 2025-03-09 | Tiny Wan decoder exists | No reviewed MLX port or comparable preservation claim; output-changing substitution |
| `Blaizzy/mlx-video` | Pinned by LiveFrame | Video generation on MLX exists | Existence is not a comparable controlled benchmark |
| `dgrauet/ltx-2-mlx` | 2026 | MLX LTX-2 memory engineering and cache controls | Different model/workload; one reviewed cache-limit comparison reported little wall-time change |
| Apple MLX `set_cache_limit` documentation | Current | Official free-buffer cache control mechanism | Documents behavior, not LiveFrame’s video timing result |
| *Benchmarking On-Device ML on Apple Silicon with MLX*, arXiv:2510.18921 | 2025 | Academic MLX benchmarks | Transformer/NLP scope, not video VAE decoding |

### Paper-safe novelty statement

> Prior work establishes video VAE decoding as an important bottleneck after DiT acceleration and documents MLX cache controls for memory management. In the reviewed primary sources, we did not find a prior public report combining an Apple-Silicon MLX Wan workload, a measured end-to-end speedup from free-buffer-cache policy alone, and repeated byte-identical pre-encode RGB digest records. This statement is limited to the measured configuration and preservation criterion.

Do not claim fastest Apple-Silicon video generation or universal priority without a standardized cross-system benchmark.

## Periodic refresh K/V reuse

### Closest sources

| Source | Date | Mechanism | Distinction from LiveFrame |
| --- | --- | --- | --- |
| Gao et al., *Ca2-VDM*, arXiv:2411.16375 | 2024; ICML 2025 reported by authors | Causal historical-frame K/V reuse and denoising-step cache sharing | Current target K/V are computed; no anchor content substituted for future-frame projections |
| Zeng et al., *X-Cache*, arXiv:2604.20289 | 2026-04 | Cross-chunk residual reuse with unconditional full compute on persistent clean-KV-writing chunks | Reuses block residuals, not raw normalized K/V; its clean-write protection is corroborating design evidence |
| Zheng et al., *Causal-rCM*, arXiv:2606.25473 | 2026-06 | Autoregressive distillation and optional noisy-context cache reuse | Each chunk contributes its own trained state; no older anchor impersonates current-frame K/V |
| *MemRoPE*, arXiv:2603.12513 | 2026 | Unrotated historical K/V and RoPE applied online | Historical memory remains historical context |
| *Deep Forcing*, arXiv:2512.05081 | 2025 | Temporal RoPE realignment and history selection | Repositions retained history rather than replacing current-frame projections |
| *Anchor Forcing*, arXiv:2603.13405 | 2026 | Anchor-guided contextual re-cache and region-specific RoPE | Requires recomputation/distillation; retrieved K/V remain context |

No reviewed source tested the exact all-layer intervention used by LiveFrame: every third frame as an anchor, raw post-normalization K/V substituted for the next two frames, destination temporal re-RoPE on K, and current queries/attention/downstream computation retained.

### Paper-safe negative statement

> Under the tested 21-frame causal video-diffusion configuration, periodic raw-K/V substitution failed all 16 aggregate acceptance cells across 64 H100 runs. This rejects the specific intervention and evaluation contract; it does not establish that cross-frame K/V reuse generally fails.

Causal-rCM means “train away refresh” is not an untouched general idea. A future differentiated hypothesis would need a specific multi-frame shifted-position schedule and prospective quality contract.

## Publication impact assessment

### Why publishing is useful

- **Reproducibility:** visual autoregressive rollout is a strong setting for demonstrating that seed equality and byte-identical random inputs do not guarantee cross-runtime identity.
- **Measurement practice:** the four claim layers, prospective thresholds, long horizons, timing eligibility, and fail-closed artifacts are reusable beyond this model.
- **MLX engineering:** the M5 component budget and exact-output allocator intervention provide a concrete baseline in a sparse public benchmark surface.
- **Negative-result value:** the refresh STOP prevents others from treating positional re-RoPE alone as sufficient justification for stale content reuse.
- **Priority:** the evidence tag and arXiv version provide a public timestamp while VAE acceleration and causal-cache research are active.

### Expected limits on impact

- It is not a new generator, quality record, or real-time result.
- The clean M5 benchmark is one short fixture on one machine and model family.
- The raw RGB payload for the allocator experiment was not retained.
- The cross-runtime study does not identify the first divergent operator.
- The refresh study does not ablate stale K versus V, skipped projection/normalization, period, or layer depth.
- Tier A/B evidence must be publicly reachable for the methodology claim to carry full weight.

### Practical verdict

Publish after the author block, evidence URLs, preview content review, and arXiv metadata are complete. A realistic outcome is a durable methods/reproducibility citation and useful MLX benchmark reference, not broad headline attention. The cross-runtime result has the strongest citation potential; the measurement framework has the longest shelf life.
