import torch
from lab04_starter import RecallNet, train_recall


def verify_gate_slices(hidden=32, target_bias=3.0):
    print("=== Gate Bias Slice Verification ===")
    for kind, gate_name, n_gates in [("lstm", "forget", 4), ("gru", "update", 3)]:
        torch.manual_seed(0)
        net = RecallNet(kind, hidden=hidden, gate_bias=target_bias)
        eff_bias = (net.rnn.bias_ih_l0 + net.rnn.bias_hh_l0).detach()
        slices = [eff_bias[i * hidden:(i + 1) * hidden].mean().item() for i in range(n_gates)]
        target_slice = eff_bias[hidden:2 * hidden]
        assert torch.allclose(target_slice, torch.full_like(target_slice, target_bias)), (
            f"Slice verification failed for {kind}"
        )
        print(
            f"- `{kind.upper():4s} ({gate_name:6s} gate is slice [{hidden}:{2*hidden}]): "
            f"mean effective bias per gate block = {[round(s, 3) for s in slices]}`"
        )
    print()


def run_benchmark():
    verify_gate_slices(hidden=32, target_bias=3.0)

    configs = [
        ("Plain RNN", "rnn", None),
        ("LSTM (default)", "lstm", None),
        ("LSTM (forget bias 3)", "lstm", 3.0),
        ("GRU (default)", "gru", None),
        ("GRU (update bias 3)", "gru", 3.0),
    ]
    lengths = [10, 30, 60]
    seeds = [0, 1, 2]

    print("=== Piece 1: Delayed-Recall Benchmark (Markdown Table for README) ===")
    print("| Configuration | Length | Seed 0 | Seed 1 | Seed 2 | Mean |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    results = {}
    for label, kind, bias in configs:
        for seq_len in lengths:
            accs = []
            for seed in seeds:
                acc, _ = train_recall(
                    kind=kind,
                    seq_len=seq_len,
                    seed=seed,
                    gate_bias=bias,
                    steps=600,
                    batch_size=64,
                    lr=3e-3,
                    clip=1.0,
                )
                accs.append(acc)
            mean_acc = sum(accs) / len(accs)
            results[(label, seq_len)] = (accs, mean_acc)
            print(
                f"| {label} | {seq_len} | {accs[0]:.3f} | {accs[1]:.3f} | {accs[2]:.3f} | {mean_acc:.3f} |"
            )
    return results


if __name__ == "__main__":
    run_benchmark()