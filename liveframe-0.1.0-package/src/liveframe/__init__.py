"""LiveFrame: evidence-first measurement of real-time autoregressive diffusion video.

Real-time video generation, honestly measured.

This initial release ships the evidence verifier for LiveFrame claim ledgers
(schema ``liveframe.publication-claims.v1``): it re-hashes referenced artifacts,
checks digest bindings, and recomputes headline arithmetic from ledger values.
The streaming engine APIs (``stream``, ``clip``, ``benchmark``) land in later releases.
"""

__version__ = "0.1.0"
