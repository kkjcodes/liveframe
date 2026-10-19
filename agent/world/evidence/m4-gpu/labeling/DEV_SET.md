# Development set — DO NOT use for validation

These 64 MG2 quick-suite clips + `labels.csv` (Kumar, 2026-10-14) are the
DEVELOPMENT set for the DIS-flow control-classifier recalibration (Kumar
decision, lessons.md). Tune thresholds against these.

NEVER use them to validate/promote control metrics — that is training on the
test set. The M5 validation uses FRESH clips (all supported tokens, >=20/class,
blind, two labelers). See agent/world/m5-reference-campaign.md.

Pilot result: 82.5% overall classifier-vs-human; yaw weak (turn-left 69%);
only W/L/0 commanded (S/A/D/R untested). agreement.json has the full breakdown.
