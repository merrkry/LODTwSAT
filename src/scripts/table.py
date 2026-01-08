"""Table printing utilities for experiments."""

from typing import Any


def compute_column_widths(
    max_samples: int,
    max_features: int,
    *,
    w_rate: int = 5,
    w_cart: int = 4,
    w_dt1: int = 4,
    w_cart_test: int = 9,
    w_dt1_test: int = 9,
    w_time: int = 7,
) -> dict[str, int]:
    """Compute column widths based on max values and header lengths."""
    return {
        "rate": max(w_rate, len("0.01")),
        "n": max(len(str(max_samples)), len("n")),
        "f": max(len(str(max_features)), len("f")),
        "cart": max(w_cart, len("CART")),
        "dt1": max(w_dt1, len("DT1")),
        "cart_test": max(w_cart_test, len("CART_test")),
        "dt1_test": max(w_dt1_test, len("DT1_test")),
        "time": max(w_time, len("time")),
    }


def print_header(widths: dict[str, int]) -> None:
    """Print table header."""
    print(
        f"{'r':^{widths['rate']}} | "
        f"{'n':^{widths['n']}} | "
        f"{'f':^{widths['f']}} | "
        f"{'CART':^{widths['cart']}} | "
        f"{'DT1':^{widths['dt1']}} | "
        f"{'CART_test':^{widths['cart_test']}} | "
        f"{'DT1_test':^{widths['dt1_test']}} | "
        f"{'time':^{widths['time']}} |"
    )
    total_width = sum(widths.values()) + 7 * 3 + 8
    print("-" * total_width)


def print_batch_row(rate: float, agg: dict[str, Any], widths: dict[str, int]) -> None:
    """Print a single batch row."""
    dt1_size_str = f"{agg['dt1_size']:.1f}" if agg["dt1_size"] is not None else "-"
    dt1_test_str = (
        f"{agg['dt1_test_acc']:.2f}" if agg["dt1_test_acc"] is not None else "-"
    )
    dt1_time_str = f"{agg['dt1_time']:.4f}" if agg["dt1_time"] is not None else "-"

    cart_size_str = f"{int(agg['cart_size'])}" if agg["cart_size"] is not None else "-"
    cart_test_str = (
        f"{agg['cart_test_acc']:.2f}" if agg["cart_test_acc"] is not None else "-"
    )

    status = agg.get("status", "")

    print(
        f"{rate:>{widths['rate']}.2f} | "
        f"{agg['samples']:>{widths['n']}} | "
        f"{agg['features']:>{widths['f']}} | "
        f"{cart_size_str:>{widths['cart']}} | "
        f"{dt1_size_str:>{widths['dt1']}} | "
        f"{cart_test_str:>{widths['cart_test']}} | "
        f"{dt1_test_str:>{widths['dt1_test']}} | "
        f"{dt1_time_str:>{widths['time']}} | {status}"
    )
