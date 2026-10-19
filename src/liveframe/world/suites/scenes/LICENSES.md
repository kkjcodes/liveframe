# Suite scene provenance

Every scene in this directory is PROCEDURAL: generated deterministically by
`tools/generate_suite_scenes.py` (seeded NumPy only), stored as .npy for
byte-deterministic regeneration. No photographs, no real places, no people.
Provenance and hashes are enforced by tests/fixtures/MANIFEST.json and
tests/flywheel/test_fixtures_fictional.py, which regenerates each scene from
the generator in CI and compares hashes.
