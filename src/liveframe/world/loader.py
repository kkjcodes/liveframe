"""Resolve a user-supplied adapter spec into a live WorldBackend.

The adoption contract: you should not have to touch LiveFrame's source to
measure your own model. ``liveframe world run my_adapter.py:MyWorld`` loads
the class from your file in-process; add ``--python /path/to/model-env/bin/python``
and the same file runs bridged inside your model's environment instead, with
the harness staying in its own dependency-free process.

Spec grammar: ``FILE.py:ClassName`` (a path, resolved relative to the working
directory) or ``package.module:ClassName`` (importable module). Anything the
spec names must implement the five-method :class:`WorldBackend` protocol; a
near miss produces an error listing exactly which members are missing, not a
traceback from deep inside a run.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from liveframe.world.backend import WorldBackend

_PROTOCOL_MEMBERS = ("capabilities", "load", "reset", "step", "run_script", "close")


def parse_adapter_spec(spec: str) -> tuple[str, str]:
    """Split ``target:ClassName``; the final colon is the separator (Windows-safe)."""

    target, separator, class_name = spec.rpartition(":")
    if not separator or not target or not class_name or not class_name.isidentifier():
        raise ValueError(
            f"adapter spec must be FILE.py:ClassName or module:ClassName, got {spec!r}"
        )
    return target, class_name


def load_adapter_class(spec: str) -> type:
    target, class_name = parse_adapter_spec(spec)
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.exists():
            raise FileNotFoundError(f"adapter file not found: {path}")
        # Prefixed module name: never shadows a real installed module, and
        # registering in sys.modules keeps pickling/typing introspection of
        # adapter-defined classes working.
        module_name = f"liveframe_adapter_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not load adapter module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        loaded = getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(f"{target} has no class named {class_name!r}") from error
    if not isinstance(loaded, type):
        raise TypeError(f"{spec} names {type(loaded).__name__}, not a class")
    missing = [name for name in _PROTOCOL_MEMBERS if not hasattr(loaded, name)]
    if missing:
        raise TypeError(
            f"{class_name} does not implement the WorldBackend protocol: "
            f"missing {', '.join(missing)} "
            "(see examples/byo_adapter/ for a complete minimal adapter)"
        )
    return loaded


def load_backend(spec: str, **kwargs: Any) -> WorldBackend:
    """Instantiate the adapter named by ``spec`` and verify the protocol."""

    backend = load_adapter_class(spec)(**kwargs)
    assert isinstance(backend, WorldBackend)  # runtime_checkable; belt and braces
    return backend
