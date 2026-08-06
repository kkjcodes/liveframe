# Third-Party Assets and Reproduction Boundary

LiveFrame is licensed under Apache-2.0. This repository does not redistribute model weights or a checkout of the upstream Causal-Forcing repository. The ignored `third_party/`, `weights/`, and `checkpoints/` directories are machine-local inputs.

Reproduction requires users to acquire the following assets from their original publishers and accept the terms published there:

| Asset | Pinned source | Recorded license | Redistribution in LiveFrame |
| --- | --- | --- | --- |
| Causal-Forcing source | `thu-ml/Causal-Forcing@8db419e341e5fc52542c0b2c4542728420ddfb4a` | Apache-2.0 | Not redistributed |
| Causal-Forcing++ framewise checkpoints | `zhuhz22/Causal-Forcing@2f8eb8bb6eeb1238da9d13e5420d342a74d634a6` | Apache-2.0 metadata | Not redistributed |
| Rolling Forcing long-video checkpoint | `zhuhz22/Causal-Forcing@2f8eb8bb6eeb1238da9d13e5420d342a74d634a6` | Apache-2.0 metadata | Not redistributed |
| Wan2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | Apache-2.0 metadata and model-card use terms | Not redistributed |
| UMT5 XXL | `google/umt5-xxl@66cb9e7e85526fe440a945569e42c72fb6cbc0ad` | Apache-2.0 metadata | Not redistributed |
| mlx-video | `Blaizzy/mlx-video@87db56a51758fefb748a359b90a5283bb8ba4837` | MIT | Installed as an external dependency |

The machine-readable source revisions, expected byte sizes, and local SHA-256 anchors are in `artifacts/upstream.v1.json`. LiveFrame adapters import upstream interfaces but do not copy upstream source into the tracked package.

## Release Decision

The public repository may include LiveFrame source, tests, schemas, compact evidence JSON, generated previews, and reproduction manifests under the project license. It must not include upstream model weights or a vendored Causal-Forcing tree unless a later release separately preserves all applicable upstream licenses, notices, model-card terms, and attribution requirements.

Generated preview videos are research evidence derived from model outputs, not model weights. Their release must still follow the prompt/content review and hosting policy selected for the publication artifact.

This is an engineering inventory, not legal advice. License metadata was checked against the pinned upstream repository/model pages on 2026-08-06; release owners remain responsible for confirming terms at distribution time.
