# Motion labeling

Clips are H.264 (`avc1`) where the OpenCV build allows and play in any
browser; if the key's `codec` column says `mp4v`, open them in
QuickTime or VLC instead.

Watch each clip in `clips/` and write the motion you OBSERVE into the
`observed_motion` column using one token from the legend. Do not guess
from file order; use `?` when unsure. Prefer `labeling_sheet_blinded.csv`
(the commanded action can bias you). Do not open `labeling_key.csv`.

- `W` — camera/agent moves forward
- `S` — camera/agent moves backward
- `A` — strafes left
- `D` — strafes right
- `L` — turns (yaws) left
- `R` — turns (yaws) right
- `0` — no motion / standing still
- `?` — cannot tell / none of the above
