# LiveFrame

**Real-time video generation, honestly measured.**

LiveFrame is an evidence-first measurement framework for real-time autoregressive
diffusion video, spanning NVIDIA CUDA and Apple MLX. It separates four claim layers —
numerical trajectory, intrinsic quality, same-seed identity, and complete-wall
performance — with pre-registered thresholds and digest-bound provenance.

Paper: *LiveFrame: Evidence-First Measurement of Real-Time Autoregressive Diffusion
Video Across CUDA and Apple MLX* (2026).

## This release (0.2.0)

This release adds dependency-free orchestration APIs for measured generation:

- `stream()` yields timed native-frame events.
- `clip()` requires exact finite completion and reports complete measured wall time.
- `benchmark()` separates warmups from measured trials and reports median and p95 wall time.

Model loading remains behind a small backend protocol. The package does not bundle Wan,
MLX, or CUDA weights and does not claim to provide a production model backend yet.

```python
from collections.abc import Iterator

from liveframe import GenerationRequest, NativeFrame, benchmark, clip


class MyBackend:
	def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]:
		for index in range(request.native_frames):
			pixels = generate_one_frame(index, request)  # your model integration
			yield NativeFrame(index=index, payload=pixels)


request = GenerationRequest("a paper boat crossing a rain puddle")
result = clip(MyBackend(), request)
print(result.metrics.generated_fps)

summary = benchmark(MyBackend(), request, warmup_runs=1, trials=3)
print(summary.median_generated_fps, summary.p95_wall_seconds)
```

If a backend emits chunks, `native_frame_count` keeps generated-frame accounting
explicit:

```python
yield NativeFrame(index=0, payload=first_chunk, native_frame_count=4)
yield NativeFrame(index=4, payload=second_chunk, native_frame_count=4)
```

Missing, duplicate, out-of-order, and excess native frames fail closed. Generated FPS
is computed from the declared native-frame count and measured wall time, never playback
duration or iterator-item count.

The **evidence verifier** for LiveFrame claim ledgers remains available:

```bash
pip install liveframe

# re-hash every unique digest-bound artifact referenced by a ledger (fail-closed)
liveframe verify path/to/liveframe-publication-claims.v1.json --artifacts-root path/to/repo

# independently recompute headline arithmetic from values recorded in the ledger
liveframe recompute path/to/liveframe-publication-claims.v1.json
```

`verify` requires the separately distributed evidence artifacts referenced by the
ledger; model weights and evidence files are not bundled in this Python package.
It rejects missing files, digest mismatches, unsupported schemas, conflicting
references, and artifact paths outside the selected root.

See the repository for the full evidence bundle, claim protocol, and reports.

## License

Apache-2.0
