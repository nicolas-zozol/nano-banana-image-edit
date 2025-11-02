"""Test configuration for pytest."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ai: real API tests that may cost money")
    config.addinivalue_line("markers", "slow: tests expected to run longer than ~1 second")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_ai = os.getenv("RUN_AI_TESTS") == "1"
    skip_marker = pytest.mark.skip(reason="Set RUN_AI_TESTS=1 to run AI integration tests.")

    if run_ai:
        return

    for item in items:
        if "ai" in item.keywords:
            item.add_marker(skip_marker)

