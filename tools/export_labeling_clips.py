"""Export short clips from a world run for human motion labeling (M4 item 2).

Control metrics stay OUT of public reports until classifier-vs-human agreement
on REAL model output exists, and commanded actions are never ground truth —
the human label is. This tool closes that loop in two steps:

``export``
    Slice a run's captured frames into ~2-3 s clips aligned to commanded-action
    segments (stratified across action classes), write them as .mp4, and emit:

    - ``labeling_sheet.csv``   — clip_id, commanded_action, observed_motion
      (blank), notes (blank): the sheet shape Kumar specified.
    - ``labeling_sheet_blinded.csv`` — same rows WITHOUT the commanded action;
      recommended for labelers so the command cannot bias them.
    - ``labeling_key.csv``     — clip provenance (run, case, frame range) plus
      the classifier's majority prediction per clip. Keep away from labelers.
    - ``README.md``            — labeler instructions and the class legend.

``agreement``
    Join a filled sheet against the key and report classifier-vs-human
    agreement per class (plus human-vs-commanded, as context only).

Clip IDs are shuffled with a seeded RNG so neither file order nor ID order
leaks case structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.world.export import open_video_writer  # noqa: E402
from liveframe.world.metrics.control import (  # noqa: E402
    classify_pair,
    commanded_class,
    pair_features,
    to_grayscale,
)
from liveframe.world.sinks import load_captured_frames  # noqa: E402

CLASS_LEGEND = {
    "W": "camera/agent moves forward",
    "S": "camera/agent moves backward",
    "A": "strafes left",
    "D": "strafes right",
    "L": "turns (yaws) left",
    "R": "turns (yaws) right",
    "0": "no motion / standing still",
    "?": "cannot tell / none of the above",
}


def _case_segments(case_dir: Path) -> list[tuple[int, str, int]]:
    """Contiguous same-action frame segments from the case's timing rows."""

    rows = [
        json.loads(line)
        for line in (case_dir / "timing.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    segments: list[tuple[int, str, int]] = []
    for row in sorted(rows, key=lambda row: row["start_frame"]):
        action, start, frames = row["action"], row["start_frame"], row["frames"]
        if segments and segments[-1][1] == action:
            prev_start, _, prev_frames = segments[-1]
            segments[-1] = (prev_start, action, prev_frames + frames)
        else:
            segments.append((start, action, frames))
    return segments


def _candidate_windows(
    run_dir: Path, clip_frames: int, min_frames: int
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for case_dir in sorted((run_dir / "cases").glob("*")):
        case_file = case_dir / "case.json"
        if not case_file.exists():
            continue
        case = json.loads(case_file.read_text(encoding="utf-8"))
        if case.get("status") != "ok":
            continue
        if not (case_dir / "timing.jsonl").exists():
            continue
        for start, action, hold in _case_segments(case_dir):
            cursor = start
            while cursor + min_frames <= start + hold:
                length = min(clip_frames, start + hold - cursor)
                windows.append(
                    {
                        "run_dir": str(run_dir),
                        "case_id": case["case_id"],
                        "start_frame": cursor,
                        "end_frame": cursor + length,
                        "commanded_action": action,
                        "commanded_class": commanded_class(action) or "?",
                    }
                )
                cursor += length
    return windows


def _stratified_sample(
    windows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_class[window["commanded_class"]].append(window)
    for pool in by_class.values():
        rng.shuffle(pool)
    picked: list[dict[str, Any]] = []
    classes = sorted(by_class)
    while len(picked) < count and any(by_class[name] for name in classes):
        # Shuffle the round order so the remainder (count % classes) doesn't
        # always favor the alphabetically-first classes.
        rng.shuffle(classes)
        for name in classes:
            if by_class[name] and len(picked) < count:
                picked.append(by_class[name].pop())
    return picked


def _classifier_majority(frames: np.ndarray) -> str:
    gray = to_grayscale(frames)
    votes = [
        classify_pair(pair_features(gray[i], gray[i + 1])) for i in range(len(gray) - 1)
    ]
    if not votes:
        return "?"
    return Counter(votes).most_common(1)[0][0]


def _write_clip(frames: np.ndarray, path: Path, fps: float) -> str:
    height, width = frames.shape[1:3]
    writer, codec = open_video_writer(path, fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return codec


def _native_fps(run_dir: Path) -> float:
    for name in ("report.json", "manifest.json"):
        candidate = run_dir / name
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            capabilities = payload.get("capabilities") or {}
            if "native_fps" in capabilities:
                return float(capabilities["native_fps"])
    raise SystemExit(f"no report.json/manifest.json with capabilities in {run_dir}")


def export(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    all_windows: list[dict[str, Any]] = []
    fps_by_run: dict[str, float] = {}
    for run_dir in args.run_dirs:
        fps = _native_fps(run_dir)
        fps_by_run[str(run_dir)] = fps
        clip_frames = round(args.clip_seconds * fps)
        min_frames = round(args.min_clip_seconds * fps)
        all_windows.extend(_candidate_windows(run_dir, clip_frames, min_frames))
    if not all_windows:
        raise SystemExit("no eligible clip windows found (need ok cases with captures)")

    picked = _stratified_sample(all_windows, args.clips, rng)
    rng.shuffle(picked)  # anonymize ordering before IDs are assigned

    frames_cache: dict[tuple[str, str], np.ndarray] = {}
    key_rows: list[dict[str, Any]] = []
    for index, window in enumerate(picked):
        clip_id = f"clip-{index:03d}"
        cache_key = (window["run_dir"], window["case_id"])
        if cache_key not in frames_cache:
            frames_cache[cache_key] = load_captured_frames(
                Path(window["run_dir"]) / "cases" / window["case_id"] / "frames_256"
            )
        frames = frames_cache[cache_key][window["start_frame"] : window["end_frame"]]
        fps = fps_by_run[window["run_dir"]]
        codec = _write_clip(frames, clips_dir / f"{clip_id}.mp4", fps)
        key_rows.append(
            {
                "clip_id": clip_id,
                **window,
                "classifier_prediction": _classifier_majority(frames),
                "fps": fps,
                "codec": codec,  # avc1 plays in browsers; mp4v needs QuickTime/VLC
            }
        )

    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    sheet_rows = [
        {
            "clip_id": row["clip_id"],
            "commanded_action": row["commanded_action"],
            "observed_motion": "",
            "notes": "",
        }
        for row in key_rows
    ]
    _write_csv(
        args.output_dir / "labeling_sheet.csv",
        ["clip_id", "commanded_action", "observed_motion", "notes"],
        sheet_rows,
    )
    _write_csv(
        args.output_dir / "labeling_sheet_blinded.csv",
        ["clip_id", "observed_motion", "notes"],
        sheet_rows,
    )
    _write_csv(
        args.output_dir / "labeling_key.csv",
        [
            "clip_id",
            "run_dir",
            "case_id",
            "start_frame",
            "end_frame",
            "commanded_action",
            "commanded_class",
            "classifier_prediction",
            "fps",
            "codec",
        ],
        key_rows,
    )
    _write_labeling_page(args.output_dir, [row["clip_id"] for row in key_rows])
    legend = "\n".join(f"- `{token}` — {text}" for token, text in CLASS_LEGEND.items())
    (args.output_dir / "README.md").write_text(
        "# Motion labeling\n\n"
        "Clips are H.264 (`avc1`) where the OpenCV build allows and play in any\n"
        "browser; if the key's `codec` column says `mp4v`, open them in\n"
        "QuickTime or VLC instead.\n\n"
        "Watch each clip in `clips/` and write the motion you OBSERVE into the\n"
        "`observed_motion` column using one token from the legend. Do not guess\n"
        "from file order; use `?` when unsure. Prefer `labeling_sheet_blinded.csv`\n"
        "(the commanded action can bias you). Do not open `labeling_key.csv`.\n\n"
        f"{legend}\n",
        encoding="utf-8",
    )
    per_class = Counter(row["commanded_class"] for row in key_rows)
    print(
        json.dumps(
            {
                "clips": len(key_rows),
                "per_class": dict(sorted(per_class.items())),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


def _write_labeling_page(output_dir: Path, clip_ids: list[str]) -> None:
    """Self-contained blinded labeling UI next to the clips.

    Opens in any browser straight from disk, references ``clips/<id>.mp4`` by
    relative path (H.264 plays inline), records one motion label per clip, and
    downloads a ``labels.csv`` that the ``agreement`` subcommand reads directly.
    The commanded action is deliberately NOT embedded — labels must come from
    what the labeler sees, not from the command (M4 item 2).
    """

    buttons = "".join(
        f'<button data-label="{token}">'
        f"<b>{token}</b><span>{html_escape(text)}</span></button>"
        for token, text in CLASS_LEGEND.items()
    )
    clip_json = json.dumps(clip_ids)
    page = _LABELING_PAGE_TEMPLATE.replace("__BUTTONS__", buttons).replace(
        "__CLIPS__", clip_json
    )
    (output_dir / "labeling.html").write_text(page, encoding="utf-8")


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


_LABELING_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiveFrame motion labeling</title>
<style>
:root{color-scheme:light dark;font-family:-apple-system,system-ui,sans-serif}
body{margin:0;background:#15151a;color:#e8e8e4;display:flex;flex-direction:column;
     align-items:center;min-height:100vh}
main{width:min(640px,94vw);padding:1.25rem}
h1{font-size:1.1rem;font-weight:600}
.bar{height:6px;background:#2a2a30;border-radius:3px;overflow:hidden;margin:.5rem 0 1rem}
.bar>i{display:block;height:100%;background:#4c9e5f;width:0;transition:width .2s}
video{width:100%;border-radius:10px;background:#000;aspect-ratio:16/9}
.meta{display:flex;justify-content:space-between;color:#9a9a94;font-size:.85rem;margin:.4rem 0}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.5rem;margin:.75rem 0}
button{display:flex;flex-direction:column;align-items:flex-start;gap:.15rem;
       padding:.55rem .8rem;border:1px solid #3a3a42;border-radius:9px;background:#1f1f26;
       color:inherit;cursor:pointer;text-align:left;font:inherit}
button:hover{border-color:#4c7fb8}
button.chosen{border-color:#4c9e5f;background:#1d2b21}
button b{font-size:1.1rem}button span{font-size:.78rem;color:#9a9a94}
.nav{display:flex;gap:.5rem;margin-top:.5rem}
.nav button{flex-direction:row;justify-content:center}
.done{background:#4c9e5f;border-color:#4c9e5f;color:#06220e;font-weight:600}
.hint{color:#6b6b66;font-size:.78rem;margin-top:1rem;line-height:1.5}
textarea{width:100%;margin-top:.5rem;background:#1f1f26;color:inherit;border:1px solid #3a3a42;
         border-radius:8px;padding:.4rem;font:inherit;resize:vertical;min-height:2.2rem}
</style></head><body><main>
<h1>LiveFrame motion labeling</h1>
<div class="bar"><i id="prog"></i></div>
<video id="vid" autoplay loop muted playsinline></video>
<div class="meta"><span id="count"></span><span id="status"></span></div>
<p style="margin:.3rem 0;color:#9a9a94;font-size:.9rem">What motion do you OBSERVE in this clip?</p>
<div class="grid">__BUTTONS__</div>
<textarea id="notes" placeholder="optional note"></textarea>
<div class="nav">
  <button id="prev">&larr; Prev</button>
  <button id="next">Next &rarr;</button>
  <button id="reset">Reset all</button>
  <button id="save" class="done">Download labels.csv</button>
</div>
<p class="hint">Keys: W/A/S/D move, Q = turn left, E = turn right, 0 or space = no motion,
/ = unsure. Label every clip, then Download and send the CSV back. Your answers are the
ground truth; the clips do not reveal the commanded action on purpose.</p>
</main><script>
const CLIPS = __CLIPS__;
const KEYMAP = {w:"W",s:"S",a:"A",d:"D",q:"L",e:"R",0:"0"," ":"0","/":"?"};
const answers = {}, notes = {};
let i = 0;
const vid=document.getElementById("vid"), prog=document.getElementById("prog"),
      count=document.getElementById("count"), status=document.getElementById("status"),
      notesEl=document.getElementById("notes"),
      btns=[...document.querySelectorAll(".grid button")];
function render(){
  const id=CLIPS[i];
  vid.src="clips/"+id+".mp4";
  count.textContent=(i+1)+" / "+CLIPS.length+"  ("+id+")";
  const n=Object.keys(answers).length;
  status.textContent=n+" labeled";
  prog.style.width=(100*n/CLIPS.length)+"%";
  notesEl.value=notes[id]||"";
  btns.forEach(b=>b.classList.toggle("chosen",answers[id]===b.dataset.label));
}
function choose(label){ answers[CLIPS[i]]=label; render(); if(i<CLIPS.length-1){i++;render();} }
btns.forEach(b=>b.onclick=()=>choose(b.dataset.label));
document.onkeydown=e=>{
  if(e.target===notesEl)return;
  const k=e.key.toLowerCase();
  if(k in KEYMAP){e.preventDefault();choose(KEYMAP[k]);}
  else if(e.key==="ArrowRight"){i=Math.min(i+1,CLIPS.length-1);render();}
  else if(e.key==="ArrowLeft"){i=Math.max(i-1,0);render();}
};
notesEl.oninput=()=>{notes[CLIPS[i]]=notesEl.value;};
document.getElementById("prev").onclick=()=>{i=Math.max(i-1,0);render();};
document.getElementById("next").onclick=()=>{i=Math.min(i+1,CLIPS.length-1);render();};
document.getElementById("reset").onclick=()=>{
  if(!confirm("Clear every label and start over?"))return;
  for(const k in answers)delete answers[k];
  for(const k in notes)delete notes[k];
  i=0; render();
};
document.getElementById("save").onclick=()=>{
  const rows=[["clip_id","observed_motion","notes"]];
  for(const id of CLIPS) rows.push([id,answers[id]||"",(notes[id]||"").replace(/[\\n",]/g," ")]);
  const esc=c=>/[",]/.test(c)?'"'+c.replace(/"/g,'""')+'"':c;
  const csv=rows.map(r=>r.map(esc).join(",")).join("\\n");
  const miss=CLIPS.length-Object.keys(answers).length;
  if(miss>0 && !confirm(miss+" clip(s) still unlabeled. Download anyway?"))return;
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="labels.csv"; a.click();
};
render();
</script></body></html>
"""


def agreement(args: argparse.Namespace) -> None:
    """Score a filled sheet against the key.

    Human ``?`` (cannot tell) rows are EXCLUDED from agreement denominators
    and reported separately as ``unsure`` — "we both couldn't tell" is not
    agreement on a motion class. Per-class rows are keyed by the COMMANDED
    class (the stratification variable), stated explicitly in the key name.
    """

    def _read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    key = {row["clip_id"].strip().lower(): row for row in _read(args.key)}
    labeled = [
        row
        for row in _read(args.sheet)
        if (row.get("observed_motion") or "").strip()
    ]
    if not labeled:
        raise SystemExit("the sheet has no filled observed_motion rows yet")

    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "match": 0, "unsure": 0}
    )
    commanded_match = 0
    scored = 0
    for row in labeled:
        clip_id = (row.get("clip_id") or "").strip().lower()
        entry = key.get(clip_id)
        if entry is None:
            raise SystemExit(f"clip_id {row.get('clip_id')!r} not in the key")
        human = (row.get("observed_motion") or "").strip().upper()
        stats = per_class[entry["commanded_class"]]
        if human == "?":
            stats["unsure"] += 1
            continue
        stats["n"] += 1
        scored += 1
        if human == entry["classifier_prediction"].upper():
            stats["match"] += 1
        if human == entry["commanded_class"].upper():
            commanded_match += 1

    if scored == 0:
        raise SystemExit("every filled row is '?'; nothing to score")
    report = {
        "schema": "liveframe.world.label-agreement.v2",
        "labeled_clips": len(labeled),
        "scored_clips": scored,
        "unsure_clips": len(labeled) - scored,
        "classifier_vs_human_by_commanded_class": {
            name: {
                "n": stats["n"],
                "unsure": stats["unsure"],
                "agreement": (stats["match"] / stats["n"]) if stats["n"] else None,
            }
            for name, stats in sorted(per_class.items())
        },
        "classifier_vs_human_overall": sum(s["match"] for s in per_class.values())
        / scored,
        # Context only — commanded actions are NOT ground truth (M4 item 2).
        "human_vs_commanded_overall_context_only": commanded_match / scored,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    exporter = commands.add_parser("export", help="slice a run into labeled clips")
    exporter.add_argument("run_dirs", type=Path, nargs="+")
    exporter.add_argument("--output-dir", type=Path, required=True)
    exporter.add_argument("--clips", type=int, default=100)
    exporter.add_argument("--clip-seconds", type=float, default=2.5)
    exporter.add_argument("--min-clip-seconds", type=float, default=2.0)
    exporter.add_argument("--seed", type=int, default=0)
    exporter.set_defaults(handler=export)

    agree = commands.add_parser("agreement", help="score a filled sheet against the key")
    agree.add_argument("--sheet", type=Path, required=True)
    agree.add_argument("--key", type=Path, required=True)
    agree.add_argument("--output", type=Path, default=None)
    agree.set_defaults(handler=agreement)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
