"""
Piece 4: Limits of recurrence.
1. Times a forward + backward pass of an LSTM across sequence lengths 50, 100, 200, and 400
   (best of 3 repeats at each length) to demonstrate the sequential time bottleneck.
2. Evaluates the open-gate LSTM (forget_bias=3.0) at lengths 60, 100, and 200 (600 updates, seed 0)
   to demonstrate the path-length / fixed-state capacity limit.
"""
import time
import torch
import torch.nn as nn
from lab04_starter import N_INPUTS, train_recall


def time_lstm_pass(seq_len, batch_size=64, hidden=32, repeats=3):
    """Returns the best (minimum) time in milliseconds over `repeats` runs for a
    forward + backward pass of an LSTM at the given sequence length."""
    torch.manual_seed(0)
    lstm = nn.LSTM(N_INPUTS, hidden, batch_first=True)
    x = torch.randn(batch_size, seq_len, N_INPUTS)

    # Warmup pass
    out, _ = lstm(x)
    out.sum().backward()
    lstm.zero_grad()

    times_ms = []
    for _ in range(repeats):
        lstm.zero_grad()
        t0 = time.perf_counter()
        out, _ = lstm(x)
        loss = out.sum()
        loss.backward()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    return min(times_ms)


def run_timing():
    lengths = [50, 100, 200, 400]
    print("=== Piece 4A: LSTM Forward + Backward Pass Timing (Markdown Table) ===")
    print("| Sequence Length ($T$) | Best Time (ms) | Ratio vs. $T = 50$ |")
    print("| :---: | :---: | :---: |")

    base_time = None
    for T in lengths:
        ms = time_lstm_pass(T, batch_size=64, hidden=32, repeats=3)
        if base_time is None:
            base_time = ms
        ratio = ms / base_time
        print(f"| {T} | {ms:.2f} | {ratio:.2f}x |")

    print("\n=== Piece 4B: Distance Limit Check — Open-Gate LSTM (bias=3.0) Beyond Length 60 ===")
    print("| Sequence Length ($T$) | Validation Accuracy (Seed 0) | Outcome |")
    print("| :---: | :---: | :--- |")
    for T in [60, 100, 200]:
        acc, _ = train_recall("lstm", seq_len=T, seed=0, gate_bias=3.0, steps=600)
        outcome = "Solved" if acc > 0.9 else "Chance (0.250)"
        print(f"| {T} | {acc:.3f} | {outcome} |")


if __name__ == "__main__":
    run_timing()