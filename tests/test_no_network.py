"""The conftest.py network guard must actually stop real requests."""

import socket

import pytest
import requests


def test_sockets_are_blocked():
    with pytest.raises(RuntimeError, match="must not use the network"):
        socket.create_connection(("www.sec.gov", 443), timeout=1)


def test_requests_cannot_reach_sec():
    # Whether or not requests wraps the error, no traffic leaves the machine.
    with pytest.raises((RuntimeError, requests.ConnectionError)):
        requests.get("https://www.sec.gov/", timeout=1)


def test_heavy_ml_imports_are_blocked():
    """conftest.py refuses real imports of torch & co. (they can abort pytest on a Mac)."""
    import importlib

    from conftest import HEAVY_IMPORTS, HEAVY_MODULES

    for name in ("torch", "sentence_transformers", "transformers"):
        assert name in HEAVY_MODULES
        with pytest.raises(ModuleNotFoundError, match="must not import"):
            importlib.import_module(name)
    assert HEAVY_IMPORTS.attempts == ["torch", "sentence_transformers", "transformers"]
    HEAVY_IMPORTS.attempts.clear()  # expected here; any other test doing this fails
