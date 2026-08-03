"""Host provenance and secure mobile-operations readiness checks."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MobileReadiness:
    ssh_client: bool
    ssh_server_config: bool
    tailscale: bool
    tmux: bool
    caffeinate: bool

    @property
    def ready(self) -> bool:
        return self.ssh_client and self.tailscale and self.tmux



def mobile_readiness() -> MobileReadiness:
    return MobileReadiness(
        ssh_client=_executable("ssh") is not None,
        ssh_server_config=_executable("sshd") is not None or platform.system() == "Darwin",
        tailscale=_executable("tailscale") is not None,
        tmux=_executable("tmux") is not None,
        caffeinate=_executable("caffeinate") is not None,
    )


def collect_host_provenance() -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mobile_readiness": asdict(mobile_readiness()),
    }
    if platform.system() == "Darwin":
        provenance["macos_version"] = platform.mac_ver()[0]
        pmset = _executable("pmset")
        provenance["battery"] = _command_output([pmset, "-g", "batt"]) if pmset else None
        provenance["power_mode"] = (
            _command_output([pmset, "-g", "custom"]) if pmset else None
        )
    return provenance


def _executable(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    candidates = [
        Path("/usr/bin") / name,
        Path("/usr/sbin") / name,
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
    ]
    if name == "tailscale":
        candidates.append(Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale"))
    return next((str(path) for path in candidates if path.is_file()), None)


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None
