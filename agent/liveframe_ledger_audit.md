# LiveFrame Publication Ledger Audit

**Date:** 2026-08-06
**Ledger:** `artifacts/liveframe-publication-claims.v1.json`
**Method:** one-time manual recomputation independent of `test_publication_claims.py`

## Source artifact SHA-256

| Ledger source | Recomputed SHA-256 | Result |
| --- | --- | --- |
| M5 1 GiB baseline | `99b18d143dfff1323cdf850b367efc84b8051df676abd5fe54348b6695f0ec3d` | match |
| M5 4 GiB baseline | `5bdc9fdbae59842158f278163d738656d1e4c05996ccf7495cf4d074009e3ce1` | match |
| Sealed holdout verdict | `146cc868534f1a0955f7d8e49f899df0da63137c4ecb83acd68c1487993f3bcd` | match |
| Refresh numerical decision | `c0fe75da36e5a5e968c288021b8ca8fde8b9a1d0e71bb04ba792cc793f9bd4b4` | match |

The policy-bundle raw file SHA-256 is
`624b28e6524762cd8f19ebbc31703d3c8e1d6421aa821a9b841a95c573c258c0`.
Its canonical digest, recomputed after removing `bundle_digest`, is
`sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870`,
matching the ledger. The refresh decision canonical digest, recomputed after removing
`decision_digest`, is
`sha256:51b1146027c6213336d9e9eaf2c9bbd404e3b517ab78aad80370510f4e8624bb`,
also matching.

## Derived claims

| Quantity | Manual recomputation | Ledger |
| --- | ---: | ---: |
| Complete-wall speedup | 1.210764138475053 | match |
| Complete-wall reduction | 0.17407530647588276 | match |
| VAE-wall reduction | 0.24087729306204742 | match |
| 16 FPS target gap | 11.28424908641965 | match |

## Pixel-anchor correction

The audit found one provenance ambiguity. The `4b55...` digest is the raw RGB anchor recorded by
all three measured benchmark trials, but the 97,044,480-byte raw payload was not retained as a
file. The shipped H.264 candidate has SHA-256
`245b41d6077948e23006af753178ae943e84e0702fa423f291b66a77bfbae151`; decoding that lossy file
to RGB produces
`ee287790463104842f2a7310e2261963267f1401229fa93858d9321cc538f6c4`, not the pre-encode raw
anchor. The ledger now names `4b55...` as `recorded_raw_pixel_anchor_sha256`, states
`raw_pixel_payload_retained: false`, and separately records the verifiable encoded and decoded
video hashes.

This correction does not change the measured speedup or the trial-level exact-output result. It
does narrow reproducibility language: exact pre-encode RGB identity is supported by repeated
digest records in the immutable benchmark artifact, not by redistributing the original raw bytes.

## Audit disposition

All ledger source hashes, canonical digests, and arithmetic claims match after the provenance
field correction above. Future drift is guarded by `tests/test_publication_claims.py`.