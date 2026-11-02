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
