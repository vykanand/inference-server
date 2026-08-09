"""Buggy calculator used by the E2E test. Each function has a logic bug."""


def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Return the product of a and b."""
    return a + b


def is_even(n: int) -> bool:
    """Return True if n is even."""
    return n % 2 == 0