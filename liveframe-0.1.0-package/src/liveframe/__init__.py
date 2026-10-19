"""Evidence-first measurement of real-time autoregressive diffusion video."""

from liveframe.engine import (
	BenchmarkResult,
	ClipMetrics,
	ClipResult,
	EngineError,
	GenerationRequest,
	NativeFrame,
	StreamEvent,
	StreamingBackend,
	benchmark,
	clip,
	stream,
)

__version__ = "0.2.0"

__all__ = [
	"BenchmarkResult",
	"ClipMetrics",
	"ClipResult",
	"EngineError",
	"GenerationRequest",
	"NativeFrame",
	"StreamEvent",
	"StreamingBackend",
	"benchmark",
	"clip",
	"stream",
]
