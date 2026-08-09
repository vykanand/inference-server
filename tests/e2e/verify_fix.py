"""Correct reference implementation. The E2E test verifies the model's
edits change calculator.py to match these behaviors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

EXPECTED = {
    "add":      lambda a, b: a + b,
    "multiply": lambda a, b: a * b,
    "is_even":  lambda n: n % 2 == 0,
}


def check():
    import calculator as c
    assert c.add(2, 3) == EXPECTED["add"](2, 3)
    assert c.multiply(3, 4) == EXPECTED["multiply"](3, 4)
    assert c.is_even(2) == EXPECTED["is_even"](2)
    assert c.is_even(3) == EXPECTED["is_even"](3)
    print("MODEL FIX VERIFIED")


if __name__ == "__main__":
    check()