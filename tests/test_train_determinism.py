import random

from src.train import set_determinism


def _python_random_sequence(length: int = 5) -> list[float]:
    return [random.random() for _ in range(length)]


def test_set_determinism_resets_python_random_stream() -> None:
    set_determinism(123, deterministic=False, use_float64=False)
    first = _python_random_sequence()
    _python_random_sequence()

    set_determinism(123, deterministic=False, use_float64=False)
    second = _python_random_sequence()

    assert first == second


def test_set_determinism_changes_python_random_stream_for_new_seed() -> None:
    set_determinism(123, deterministic=False, use_float64=False)
    first = _python_random_sequence()

    set_determinism(124, deterministic=False, use_float64=False)
    second = _python_random_sequence()

    assert first != second
