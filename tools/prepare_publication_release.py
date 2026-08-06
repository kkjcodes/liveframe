"""Extract compact metrics and pre-registered preview pairs from local holdout corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def selected_rows(
    holdout: dict[str, Any], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    seeds = [str(seed) for seed in protocol["sampling"]["holdout_seeds"]]
    ranks = protocol["visual_selection"]["required_holdout_seed_ranks_per_prompt"]
    selected_seeds = {seeds[int(rank)] for rank in ranks}
    horizons = {
        int(value) for value in protocol["visual_selection"]["include_horizons_seconds"]
    }
    rows = [
        dict(row)
        for row in holdout["candidate_videos"]
        if str(row["seed_id"]) in selected_seeds
        and int(row["horizon_seconds"]) in horizons
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row["stratum"]),
            str(row["prompt_id"]),
            int(row["horizon_seconds"]),
            seeds.index(str(row["seed_id"])),
        ),
    )


def encode_preview(pixels: Path, metadata: dict[str, Any], output: Path) -> None:
    if metadata.get("dtype") != "uint8" or metadata.get("layout") != "T,H,W,C":
        raise ValueError(f"unsupported preview pixels: {pixels}")
    frames, height, width, channels = (int(value) for value in metadata["shape"])
    if channels != 3 or pixels.stat().st_size != frames * height * width * channels:
        raise ValueError(f"preview pixel geometry mismatch: {pixels}")
    if sha256_file(pixels) != metadata.get("sha256"):
        raise ValueError(f"preview pixel digest mismatch: {pixels}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "16",
            "-i",
            str(pixels),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
            "-y",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--holdout-input", type=Path, required=True)
    parser.add_argument("--h100-root", type=Path, required=True)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encode-previews", action="store_true")
    args = parser.parse_args()

    protocol = load_object(args.protocol)
    holdout = load_object(args.holdout_input)
    rows = selected_rows(holdout, protocol)
    if len(holdout["candidate_videos"]) != 128 or len(rows) != 64:
        raise ValueError("release extraction requires 128 holdout rows and 64 selected pairs")

    roots = {"cuda": args.h100_root, "mlx": args.mlx_root}
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    for backend, root in roots.items():
        result_paths = sorted(root.glob("*/result.json"))
        if len(result_paths) != 128:
            raise ValueError(f"{backend} release extraction requires 128 results")
        for source in result_paths:
            destination = args.output_root / "metrics" / backend / f"{source.parent.name}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metrics.append({"backend": backend, **artifact(destination, args.output_root)})

    previews: list[dict[str, Any]] = []
    for row in rows:
        identifiers = {
            "cuda": str(row["reference_run_id"]),
            "mlx": str(row["video_id"]),
        }
        pair_id = (
            f"{row['prompt_id']}-{row['horizon_seconds']}s-seed-{row['seed_id']}"
        )
        for backend, run_id in identifiers.items():
            source_root = roots[backend] / run_id
            result = load_object(source_root / "result.json")
            if (
                str(result["prompt_id"]) != str(row["prompt_id"])
                or int(result["horizon_seconds"]) != int(row["horizon_seconds"])
                or str(result["seed"]) != str(row["seed_id"])
            ):
                raise ValueError(f"preview result does not match holdout row: {run_id}")
            output = args.output_root / "previews" / pair_id / f"{backend}.mp4"
            if args.encode_previews:
                encode_preview(
                    source_root / "pixels.bin",
                    load_object(source_root / "pixels.bin.json"),
                    output,
                )
            preview: dict[str, Any] = {
                "backend": backend,
                "horizon_seconds": int(row["horizon_seconds"]),
                "pair_id": pair_id,
                "prompt_id": str(row["prompt_id"]),
                "run_id": run_id,
                "seed": str(row["seed_id"]),
                "source_pixels_sha256": sha256_file(source_root / "pixels.bin"),
            }
            if output.exists():
                preview.update(artifact(output, args.output_root))
            previews.append(preview)

    ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    manifest = {
        "schema": "liveframe.publication-release-tiers.v1",
        "source_protocol_sha256": sha256_file(args.protocol),
        "source_holdout_input_sha256": sha256_file(args.holdout_input),
        "visual_selection": protocol["visual_selection"],
        "tier_a_compact_metrics": {
            "files": metrics,
            "result_count": len(metrics),
        },
        "tier_b_representative_previews": {
            "encoded": args.encode_previews,
            "ffmpeg": ffmpeg_version,
            "pair_count": len(rows),
            "files": previews,
        },
        "tier_c_cold_archive": {
            "local": [
                {
                    "path": root.as_posix(),
                    "bytes": directory_bytes(root),
                    "result_count": len(list(root.glob("*/result.json"))),
                }
                for root in roots.values()
            ],
            "refresh_network_corpus": {
                "status": "network-volume-unreachable-after-compute-pod-shutdown",
                "last_validated_size_gib_approx": 93,
                "required_action": (
                    "mount on a temporary low-cost pod, copy to cold archive, verify, "
                    "then terminate the volume"
                ),
            },
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "metrics": len(metrics),
                "preview_pairs": len(rows),
                "previews_encoded": args.encode_previews,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()