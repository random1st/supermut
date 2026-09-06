"""Benchmark target: functions with deliberately incomplete tests.

Each function has KNOWN test holes (documented in run_bench.py:KNOWN_HOLES).
A mutation tool scores by producing SURVIVING mutants that point at those
holes. Do not "fix" the tests — the holes are the benchmark.
"""


def clamp(x, lo, hi):
    # holes: boundaries (x == lo, x == hi) untested
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def is_even(n):
    # hole: odd numbers untested
    return n % 2 == 0


def sign(x):
    # hole: zero untested
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def find_index(items, target):
    # hole: not-found path untested
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1


def discount(price, rate):
    # holes: rate == 1 boundary, zero price untested
    if rate < 0 or rate > 1:
        raise ValueError("rate must be in [0, 1]")
    return price * (1 - rate)


def merge_ranges(a_start, a_end, b_start, b_end):
    # hole: touching ranges (a_end == b_start) untested
    if a_end < b_start or b_end < a_start:
        return None
    return (min(a_start, b_start), max(a_end, b_end))


def truncate(text, limit):
    # holes: exact-limit boundary, ellipsis content untested
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def safe_div(a, b):
    # hole: b == 0 path untested
    if b == 0:
        return 0.0
    return a / b
