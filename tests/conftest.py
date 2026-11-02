"""Test-wide safety nets.

Every test runs with:
- no network: any attempt to resolve a hostname or open a socket connection fails
  immediately, so no test can ever reach sec.gov (or anything else) by accident;
- no SEC_USER_AGENT inherited from the developer's shell, and the current directory
  set to an empty temp folder, so neither a real env var nor a real `.env` leaks in.
  Tests that need a user agent set one explicitly with monkeypatch.setenv.
"""

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to use the real network."""


def _blocked(*args, **kwargs):
    raise NetworkBlockedError(f"tests must not use the network (attempted: {args!r})")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path_factory):
    # setenv first so monkeypatch records the original state; then a value that a test
    # loads from a .env file (load_dotenv writes straight to os.environ) is undone too.
    monkeypatch.setenv("SEC_USER_AGENT", "placeholder")
    monkeypatch.delenv("SEC_USER_AGENT")
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
