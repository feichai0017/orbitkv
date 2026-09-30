"""Test listener allocation on hosts with interface-specific services."""

import socket
from unittest.mock import Mock

from tests.support import cache_manager


def test_listener_port_rejects_another_interface(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied_port = cache_manager.find_available_port()
        occupied.bind(("127.0.0.2", occupied_port))
        occupied.listen(1)
        free_port = cache_manager.find_available_port()
        monkeypatch.setattr(
            cache_manager.secrets,
            "randbelow",
            Mock(side_effect=[15000 - 1024, 16999 - 1024, occupied_port - 1024, free_port - 1024]),
        )

        selected = cache_manager.find_available_port()

        assert selected == free_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("0.0.0.0", selected))
            listener.listen(1)
