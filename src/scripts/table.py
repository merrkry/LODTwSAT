"""Table printing utilities for experiments."""

from typing import Any


def compute_column_widths(
    max_samples: int,
    max_features: int,
    *,
    w_rate: int = 5,
    w_cart: int = 5,
    w_dt1: int = 5,
    w_cart_acc: int = 8,
    w_dt1_acc: int = 8,
    w_time: int = 7,
) -> dict[str, int]:
    """Compute column widths based on max values and header lengths."""
    return {
        "rate": max(w_rate, len("0.01")),
        "n": max(len(str(max_samples)), len("n")),
        "f": max(len(str(max_features)), len("f")),
        "cart": max(w_cart, len("n_CART")),
        "dt1": max(w_dt1, len("n_DT1")),
        "cart_acc": max(w_cart_acc, len("CART_acc")),
        "dt1_acc": max(w_dt1_acc, len("DT1_acc")),
        "time": max(w_time, len("time")),
    }


def print_header(widths: dict[str, int]) -> None:
    """Print table header."""
    print(
        f"{'r':^{widths['rate']}} | "
        f"{'n':^{widths['n']}} | "
        f"{'f':^{widths['f']}} | "
        f"{'n_CART':^{widths['cart']}} | "
        f"{'n_DT1':^{widths['dt1']}} | "
        f"{'CART_acc':^{widths['cart_acc']}} | "
        f"{'DT1_acc':^{widths['dt1_acc']}} | "
        f"{'time':^{widths['time']}} |"
    )
    total_width = sum(widths.values()) + 7 * 3 + 8
    print("-" * total_width)


def print_batch_row(rate: float, agg: dict[str, Any], widths: dict[str, int]) -> None:
    """Print a single batch row."""
    dt1_size_str = f"{agg['dt1_size']:.1f}" if agg["dt1_size"] is not None else "-"
    dt1_acc_str = (
        f"{agg['dt1_acc']:.4f}" if agg["dt1_acc"] is not None else "-"
    )
    dt1_time_str = f"{agg['dt1_time']:.4f}" if agg["dt1_time"] is not None else "-"

    cart_size_str = f"{int(agg['cart_size'])}" if agg["cart_size"] is not None else "-"
    cart_acc_str = (
        f"{agg['cart_acc']:.4f}" if agg["cart_acc"] is not None else "-"
    )

    status = agg.get("status", "")

    print(
        f"{rate:>{widths['rate']}.2f} | "
        f"{agg['samples']:>{widths['n']}} | "
        f"{agg['features']:>{widths['f']}} | "
        f"{cart_size_str:>{widths['cart']}} | "
        f"{dt1_size_str:>{widths['dt1']}} | "
        f"{cart_acc_str:>{widths['cart_acc']}} | "
        f"{dt1_acc_str:>{widths['dt1_acc']}} | "
        f"{dt1_time_str:>{widths['time']}} | {status}"
    )
