"""
Piece 3: Two controlled fixes at sequence length 30 (seeds 0, 1, 2).
1. Forget-bias sweep for LSTM: gate_bias in {0.0, 1.0, 2.0, 3.0}.
2. Gradient clipping comparison for Plain RNN: clip=1.0 (on) vs. clip=None (off).
"""
from lab04_starter import train_recall


def run_fixes():
    seeds = [0, 1, 2]
    seq_len = 30

    print("=== Piece 3A: LSTM Forget-Gate Bias Sweep at Length 30 (Markdown Table) ===")
    print("| Forget Gate Bias | Seed 0 | Seed 1 | Seed 2 | Mean |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    for bias in [0.0, 1.0, 2.0, 3.0]:
        accs = []
        for s in seeds:
            acc, _ = train_recall(
                kind="lstm", seq_len=seq_len, seed=s, gate_bias=bias,
                steps=600, batch_size=64, lr=3e-3, clip=1.0
            )
            accs.append(acc)
        mean_acc = sum(accs) / len(accs)
        label = f"`bias = {bias:.1f}` (default)" if bias == 0.0 else f"`bias = {bias:.1f}`"
        print(f"| {label} | {accs[0]:.3f} | {accs[1]:.3f} | {accs[2]:.3f} | {mean_acc:.3f} |")

    print("\n=== Piece 3B: Plain RNN Clipping Comparison at Length 30 (Markdown Table) ===")
    print("| Clipping Setting | Seed 0 | Seed 1 | Seed 2 | Mean |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    for label, clip_val in [("Clipping ON (`clip=1.0`)", 1.0), ("Clipping OFF (`clip=None`)", None)]:
        accs = []
        for s in seeds:
            acc, _ = train_recall(
                kind="rnn", seq_len=seq_len, seed=s, gate_bias=None,
                steps=600, batch_size=64, lr=3e-3, clip=clip_val
            )
            accs.append(acc)
        mean_acc = sum(accs) / len(accs)
        print(f"| {label} | {accs[0]:.3f} | {accs[1]:.3f} | {accs[2]:.3f} | {mean_acc:.3f} |")


if __name__ == "__main__":
    run_fixes()