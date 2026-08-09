"""Unit tests for calculator.py. Expected to fail until bugs are fixed."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calculator import add, multiply, is_even


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0


def test_is_even():
    assert is_even(2) is True
    assert is_even(3) is False


if __name__ == "__main__":
    test_add()
    test_multiply()
    test_is_even()
    print("ALL TESTS PASSED")