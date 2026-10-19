"""Session-wide test guards (addendum A section 6).

Outbound network access is blocked for the whole test session: tests must run
on fictional, local data only. Loopback stays allowed (multiprocessing and
local tooling may use it). GPU-marked session-setup tests that genuinely need
the network must opt in explicitly via the ``allow_network`` fixture.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_CONNECT = socket.socket.connect
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _is_loopback(address: object) -> bool:
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        return address[0] in _LOOPBACK
    # Non-tuple addresses are AF_UNIX paths: local by definition.
    return isinstance(address, (str, bytes))


def _blocked_connect(self: socket.socket, address: object) -> None:
    if _is_loopback(address):
        _REAL_CONNECT(self, address)  # type: ignore[arg-type]
        return
    raise RuntimeError(f"network access is blocked during tests (attempted {address!r})")


def _blocked_create_connection(address: object, *args: object, **kwargs: object) -> object:
    if _is_loopback(address):
        return _REAL_CREATE_CONNECTION(address, *args, **kwargs)  # type: ignore[arg-type]
    raise RuntimeError(f"network access is blocked during tests (attempted {address!r})")


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> Iterator[None]:
    socket.socket.connect = _blocked_connect  # type: ignore[method-assign, assignment]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT  # type: ignore[method-assign]
        socket.create_connection = _REAL_CREATE_CONNECTION


@pytest.fixture
def allow_network() -> Iterator[None]:
    """Explicit opt-in for @pytest.mark.gpu session-setup tests that fetch."""

    socket.socket.connect = _REAL_CONNECT  # type: ignore[method-assign]
    socket.create_connection = _REAL_CREATE_CONNECTION
    try:
        yield
    finally:
        socket.socket.connect = _blocked_connect  # type: ignore[method-assign, assignment]
        socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
