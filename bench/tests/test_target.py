"""Deliberately holey tests — see target_module docstring. Do not improve."""

from target_module import (
    clamp,
    discount,
    find_index,
    is_even,
    merge_ranges,
    safe_div,
    sign,
    truncate,
)


def test_clamp_inside():
    assert clamp(5, 0, 10) == 5


def test_clamp_below():
    assert clamp(-5, 0, 10) == 0


def test_is_even_even():
    assert is_even(4)


def test_sign_positive():
    assert sign(3) == 1


def test_sign_negative():
    assert sign(-3) == -1


def test_find_index_found():
    assert find_index(["a", "b", "c"], "b") == 1


def test_discount_half():
    assert discount(100, 0.5) == 50


def test_discount_invalid():
    try:
        discount(100, 2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_merge_overlapping():
    assert merge_ranges(0, 5, 3, 8) == (0, 8)


def test_merge_disjoint():
    assert merge_ranges(0, 2, 5, 8) is None


def test_truncate_short():
    assert truncate("hi", 10) == "hi"


def test_truncate_long():
    assert truncate("hello world", 5).startswith("hello")


def test_safe_div():
    assert safe_div(6, 3) == 2
