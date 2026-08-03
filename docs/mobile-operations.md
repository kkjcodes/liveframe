# Mobile operations runbook

## Threat model

The phone is an authenticated remote terminal, not a trusted public control server. LiveFrame assumes the host is reachable only through a private Tailscale network and SSH accepts keys only. A stolen phone or key is handled by revoking the Tailscale device and SSH key.

## Host setup checklist

- Install Tailscale and require device approval in the tailnet.
- Enable Remote Login only for the dedicated operator account.
- Add the phone client's public key to `authorized_keys`.
- Disable SSH password and keyboard-interactive authentication.
- Do not open router ports or publish the runner through a tunnel.
- Restrict the runtime directory to the operator account.
- Keep experiments in `liveframe submit`; use `tmux` only for interactive maintenance.

## Mac benchmark checklist

Before a declared benchmark, record charger state, battery level, macOS power mode, display/lid state, and ambient conditions. Prevent automatic sleep only for the run's lifetime. Remote-control overhead and setup are outside inference timing, while prompt submission and generation remain inside their declared latency boundaries.

## Recovery

A phone disconnect has no effect on a detached run. Reconnect and use `liveframe status RUN_ID`, then inspect logs. `liveframe cancel RUN_ID` first requests cooperative termination; the supervisor escalates after ten seconds. Completed and partial artifacts remain in the run directory and are listed with SHA-256 checksums.

Every submission, state transition, signal, cancellation, and terminal outcome is appended to the run's event stream. Preserve the manifest and event stream with published artifacts.
