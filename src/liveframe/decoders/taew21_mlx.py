"""MLX implementation of the pinned base TAEW2.1 causal decoder."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from liveframe.converters.taew21_to_mlx import EXPECTED_DECODER_KEYS
from liveframe.decoder_contract import DecoderChunk, DecoderRequest
from liveframe.manifests import file_sha256

MEMORY_LAYERS = (3, 4, 5, 9, 10, 11, 15, 16, 17)
FP32_MEMORY_LAYERS = (11, 17)
TEMPORAL_GROWTH = {7: 1, 13: 2, 19: 2}
SPATIAL_UPSAMPLE_LAYERS = (6, 12, 18)
EXPECTED_TEMPORAL_UPSCALE = 4
STARTUP_FRAMES_TO_TRIM = 3
CONVOLUTION_MODES = ("native", "fp32-im2col")


class TAEW21MLXDecoder:
    """Decode Wan 2.1 latents with converted TAEW2.1 decoder weights."""

    def __init__(
        self,
        weights_path: Path,
        *,
        expected_sha256: str,
        convolution_mode: str = "native",
    ) -> None:
        if convolution_mode not in CONVOLUTION_MODES:
            raise ValueError(f"unsupported TAEW2.1 convolution mode: {convolution_mode}")
        actual = file_sha256(weights_path)
        if actual != expected_sha256:
            raise ValueError(
                f"TAEW2.1 MLX weight checksum mismatch: expected {expected_sha256}, "
                f"computed {actual}"
            )
        self.mx = importlib.import_module("mlx.core")
        loaded = self.mx.load(str(weights_path))
        if not isinstance(loaded, Mapping):
            raise ValueError("TAEW2.1 MLX checkpoint must contain named tensors")
        self.weights = dict(loaded)
        if set(self.weights) != EXPECTED_DECODER_KEYS:
            missing = sorted(EXPECTED_DECODER_KEYS - set(self.weights))
            unexpected = sorted(set(self.weights) - EXPECTED_DECODER_KEYS)
            raise ValueError(
                f"TAEW2.1 MLX weight inventory mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.weights_path = weights_path
        self.weights_sha256 = actual
        self.convolution_mode = convolution_mode

    def _im2col_convolution(self, value: Any, name: str) -> Any:
        weight = self.weights[f"{name}.weight"].astype(self.mx.float32)
        bias = self.weights.get(f"{name}.bias")
        if bias is not None:
            bias = bias.astype(self.mx.float32)
        height, width = int(value.shape[1]), int(value.shape[2])
        padded = self.mx.pad(value, [(0, 0), (1, 1), (1, 1), (0, 0)])
        patches = self.mx.concatenate(
            [
                padded[:, row : row + height, column : column + width, :]
                for row in range(3)
                for column in range(3)
            ],
            axis=-1,
        )
        flattened_weight = weight.reshape(int(weight.shape[0]), -1).transpose(1, 0)
        output = patches @ flattened_weight
        return output if bias is None else output + bias

    def _convolution(
        self,
        value: Any,
        name: str,
        *,
        padding: int,
        accumulation_dtype: Any | None = None,
    ) -> Any:
        weight_name = f"{name}.weight"
        if weight_name not in self.weights:
            raise ValueError(f"TAEW2.1 MLX checkpoint is missing {weight_name}")
        if self.convolution_mode == "fp32-im2col" and padding == 1:
            if value.dtype != self.mx.float32:
                raise ValueError("TAEW2.1 fp32-im2col mode requires float32 activations")
            return self._im2col_convolution(value, name)
        weight = self.weights[weight_name]
        if accumulation_dtype is not None:
            value = value.astype(accumulation_dtype)
            weight = weight.astype(accumulation_dtype)
        output = self.mx.conv2d(value, weight, padding=padding)
        bias = self.weights.get(f"{name}.bias")
        if bias is not None and accumulation_dtype is not None:
            bias = bias.astype(accumulation_dtype)
        return output if bias is None else output + bias

    def _memory_block(self, value: Any, prior: Any | None, layer: int) -> Any:
        output_dtype = value.dtype
        accumulation_dtype = self.mx.float32 if layer in FP32_MEMORY_LAYERS else None
        compute_value = (
            value.astype(accumulation_dtype)
            if accumulation_dtype is not None
            else value
        )
        memory = self.mx.zeros_like(value) if prior is None else prior
        if accumulation_dtype is not None:
            memory = memory.astype(accumulation_dtype)
        output = self.mx.concatenate([compute_value, memory], axis=-1)
        output = self.mx.maximum(
            self._convolution(
                output,
                f"decoder.{layer}.conv.0",
                padding=1,
                accumulation_dtype=accumulation_dtype,
            ),
            0,
        )
        output = self.mx.maximum(
            self._convolution(
                output,
                f"decoder.{layer}.conv.2",
                padding=1,
                accumulation_dtype=accumulation_dtype,
            ),
            0,
        )
        output = self._convolution(
            output,
            f"decoder.{layer}.conv.4",
            padding=1,
            accumulation_dtype=accumulation_dtype,
        )
        output = self.mx.maximum(output + compute_value, 0)
        return output.astype(output_dtype) if accumulation_dtype is not None else output

    def _apply_layer(
        self,
        values: list[Any],
        layer: int,
        memory: dict[int, Any],
        call_counts: dict[int, int] | None = None,
        capture: Callable[[int, int, Any], None] | None = None,
    ) -> list[Any]:
        def record(output: Any) -> None:
            if call_counts is None:
                return
            call = call_counts.get(layer, 0)
            if capture is not None:
                capture(layer, call, output)
            call_counts[layer] = call + 1

        if layer == 0:
            outputs = []
            for value in values:
                output = self.mx.tanh(value / 3) * 3
                record(output)
                outputs.append(output)
            return outputs
        if layer in MEMORY_LAYERS:
            outputs = []
            for value in values:
                output = self._memory_block(value, memory.get(layer), layer)
                record(output)
                outputs.append(output)
                memory[layer] = value
            return outputs
        if layer in SPATIAL_UPSAMPLE_LAYERS:
            outputs = []
            for value in values:
                output = self.mx.repeat(self.mx.repeat(value, 2, axis=1), 2, axis=2)
                record(output)
                outputs.append(output)
            return outputs
        if layer in TEMPORAL_GROWTH:
            stride = TEMPORAL_GROWTH[layer]
            outputs = []
            for value in values:
                grown = self._convolution(value, f"decoder.{layer}.conv", padding=0)
                split = self.mx.split(grown, stride, axis=-1)
                record(self.mx.concatenate(split, axis=0))
                outputs.extend(split)
            return outputs
        if layer in (2, 21):
            outputs = []
            for value in values:
                output = self.mx.maximum(value, 0)
                record(output)
                outputs.append(output)
            return outputs
        if layer in (1, 8, 14, 20, 22):
            outputs = []
            for value in values:
                output = self._convolution(value, f"decoder.{layer}", padding=1)
                record(output)
                outputs.append(output)
            return outputs
        raise RuntimeError(f"unsupported TAEW2.1 decoder layer {layer}")

    def _decode_latent(
        self,
        latent: Any,
        memory: dict[int, Any],
        call_counts: dict[int, int] | None = None,
        capture: Callable[[int, int, Any], None] | None = None,
    ) -> list[Any]:
        values = [latent]
        for layer in range(23):
            values = self._apply_layer(values, layer, memory, call_counts, capture)
        return [self.mx.clip(value, 0, 1) for value in values]

    def decode_float(
        self,
        latents: Any,
        *,
        capture: Callable[[int, int, Any], None] | None = None,
    ) -> tuple[Any, dict[int, int]]:
        """Decode all frames as NFHWC float values while preserving causal execution."""

        if len(latents.shape) != 5 or tuple(int(size) for size in latents.shape[:3]) != (
            1,
            21,
            16,
        ):
            raise ValueError("TAEW2.1 parity latents must have shape 1,21,16,H,W")
        memory: dict[int, Any] = {}
        call_counts: dict[int, int] = {}
        outputs = []
        for frame in range(21):
            latent = latents[:, frame].transpose(0, 2, 3, 1)
            outputs.extend(self._decode_latent(latent, memory, call_counts, capture))
        outputs = outputs[STARTUP_FRAMES_TO_TRIM:]
        if len(outputs) != 81:
            raise ValueError(f"TAEW2.1 float decode produced {len(outputs)} frames")
        return self.mx.stack(outputs, axis=1), call_counts

    def decode(self, latents: Any, request: DecoderRequest) -> Iterator[DecoderChunk]:
        """Decode B,F,C,H,W latents and emit exact uint8 THWC native chunks."""

        import numpy as np

        if len(latents.shape) != 5 or int(latents.shape[0]) != 1:
            raise ValueError("TAEW2.1 latents must have shape B,F,C,H,W with B=1")
        if int(latents.shape[1]) != request.latent_frames or int(latents.shape[2]) != 16:
            raise ValueError("TAEW2.1 latent frame or channel count does not match request")
        expected_frames = request.latent_frames * EXPECTED_TEMPORAL_UPSCALE - STARTUP_FRAMES_TO_TRIM
        if request.native_frames != expected_frames:
            raise ValueError(
                f"TAEW2.1 request requires {expected_frames} native frames, "
                f"observed {request.native_frames}"
            )
        memory: dict[int, Any] = {}
        emitted = 0
        decoded_seen = 0
        for frame in range(request.latent_frames):
            latent = latents[:, frame].transpose(0, 2, 3, 1)
            decoded = self._decode_latent(latent, memory)
            decoded_seen += len(decoded)
            if decoded_seen <= STARTUP_FRAMES_TO_TRIM:
                continue
            if frame == 0:
                decoded = decoded[STARTUP_FRAMES_TO_TRIM:]
            value = self.mx.stack(decoded, axis=1)
            self.mx.eval(value)
            pixels = np.asarray(value[0] * 255.0).round().clip(0, 255).astype(np.uint8)
            count = int(pixels.shape[0])
            yield DecoderChunk(
                start_index=emitted,
                payload=pixels,
                shape=tuple(int(size) for size in pixels.shape),
                dtype="uint8",
                finite=bool(np.isfinite(pixels).all()),
                minimum=float(pixels.min()),
                maximum=float(pixels.max()),
            )
            emitted += count