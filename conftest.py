import re

import pytest


def extract_test_number(name: str) -> int | None:
    match = re.match(r"test_(\d+)_", name)
    if not match:
        return None
    return int(match.group(1))


def pytest_addoption(parser):
    parser.addoption("--stop-at", default=None, help="Stop after this test number (e.g. --stop-at 15)")


def pytest_configure(config):
    for i in range(1, 9):
        config.addinivalue_line("markers", f"phase{i}: test phase {i}")


def pytest_runtest_setup(item):
    stop_at = item.config.getoption("--stop-at")
    if stop_at:
        test_num = extract_test_number(item.name)
        if test_num and test_num > int(stop_at):
            pytest.skip(f"Skipping (--stop-at {stop_at})")
