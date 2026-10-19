# arXiv Submission Metadata

## Title

LiveFrame: Evidence-First Measurement of Real-Time Autoregressive Diffusion Video Across CUDA and Apple MLX

## Authors

Kumar K. Jha (Independent Researcher)

## Affiliation

Independent Researcher

## Contact email

kumar.krishnanand@gmail.com

## Primary category

`cs.CV` - Computer Vision and Pattern Recognition

## Suggested cross-list

`cs.LG` - Machine Learning

## Abstract

Claims of real-time video generation often combine unlike quantities: short-burst transformer throughput, interpolated display rate, intrinsic visual quality, and reproducibility across runtimes. We present LiveFrame, an evidence-first evaluation framework that separates four claim layers: numerical trajectory, intrinsic quality, same-seed identity, and complete-wall performance. The framework pre-registers prompts, seeds, horizons, thresholds, and aggregation; serializes stochastic inputs for literal byte-identity checks; binds model, runtime, metric, and output provenance by digest; and evaluates 15- and 60-second autoregressive horizons.

We apply this methodology to Wan2.1-T2V-1.3B derivatives on NVIDIA H100 CUDA and Apple M5 Max MLX. First, increasing the bounded MLX free-buffer cache from 1 to 4 GiB reduces complete wall time for an 81-frame, 480x832 one-step Causal Forcing++ fixture from 69.167 to 57.127 seconds, a 1.2108x speedup, while candidate and diagnostic trials reproduce the recorded pre-encode RGB anchor with zero swap. The optimized 1.418 fixture FPS remains 11.28x below our declared 16-FPS real-time target and is not a 60-second result. Second, a sealed CUDA-to-MLX Rolling Forcing holdout with byte-identical stochastic inputs passes same-seed identity for 0/128 videos, although intrinsic quality passes for 90/128. Third, a prospective 64-video H100 campaign rejects a specific periodic clean-refresh optimization: substituting re-positioned anchor keys and values for two frames after every third-frame anchor passes 0/16 aggregate quality cells. These are configuration- and mechanism-specific findings, not hardware impossibility results. Together they show why speed, quality, identity, and provenance must be reported independently in autoregressive video systems.

## Suggested comments

6 pages, 3 tables. Evidence-first systems study with one exact-output Apple MLX optimization and three bounded negative results. Code and evidence: https://github.com/kkjcodes/liveframe

## Suggested keywords

video diffusion; autoregressive generation; reproducibility; numerical divergence; Apple MLX; CUDA; systems measurement; negative results; KV cache; VAE decoding

## Submission checklist

- Confirm `Independent Researcher` as the final affiliation.
- Select an arXiv distribution license.
- Confirm the durable frozen-evidence URL resolves without authentication: https://github.com/kkjcodes/liveframe/tree/evidence-freeze-2026-08-06
- Complete content review of the 128 representative preview clips.
- Literature search refreshed on 17 September 2026; repeat only if submitting later.
- Upload `paper/liveframe-arxiv-source-2026-09-17.tar.gz` and inspect arXiv's generated PDF.
- Compare arXiv's PDF against `paper/arxiv/main.pdf` page by page.
