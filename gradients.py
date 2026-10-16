"""
Piece 2: Gradient-through-time measurement at initialization.
Measures the gradient L2 norm reaching each input step of a 60-step sequence
(hidden=32, batch=16, input_features=8) when the loss reads only the last step,
taking the median across 5 seeds (0..4) and saving a log-scale plot.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def measure_input_gradients(kind, gate_bias=None, seq_len=60, batch_size=16,
                            n_inputs=8, hidden=32, seeds=(0, 1, 2, 3, 4)):
    """Returns a tensor of shape (seq_len,) indexed by steps_back = 0 .. seq_len-1,
    representing the median L2 norm of dL/dx_t across the given seeds."""
    all_seed_norms = []

    for seed in seeds:
        torch.manual_seed(seed)
        cell_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        rnn = cell_cls(n_inputs, hidden, batch_first=True)
        head = nn.Linear(hidden, 4)

        if gate_bias is not None and kind in ("lstm", "gru"):
            with torch.no_grad():
                rnn.bias_ih_l0[hidden:2 * hidden].fill_(gate_bias / 2)
                rnn.rnn_bias_set = True
                rnn.bias_hh_l0[hidden:2 * hidden].fill_(gate_bias / 2)

        gen = torch.Generator().manual_seed(seed + 500)
        x = torch.randn(batch_size, seq_len, n_inputs, generator=gen, requires_grad=True)
        targets = torch.randint(0, 4, (batch_size,), generator=gen)

        out, _ = rnn(x)
        logits = head(out[:, -1])  # Loss reads ONLY the last step (t = seq_len - 1)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Norm of gradient per time step t = 0 .. seq_len - 1
        # Flip so index k = 0 is 0 steps back (t = 59) and k = 59 is 59 steps back (t = 0)
        step_norms = x.grad.detach().norm(dim=(0, 2))  # shape: (seq_len,)
        steps_back_norms = torch.flip(step_norms, dims=[0])
        all_seed_norms.append(steps_back_norms)

    stacked = torch.stack(all_seed_norms, dim=0)  # (n_seeds, seq_len)
    return stacked.median(dim=0).values


def run_gradients():
    seq_len = 60
    seeds = (0, 1, 2, 3, 4)
    configs = [
        ("Plain RNN", "rnn", None, "#8A6A18"),
        ("LSTM (default, bias 0)", "lstm", None, "#1C3D6E"),
        ("LSTM (forget bias 3)", "lstm", 3.0, "#0E6B57"),
    ]

    report_steps = [0, 9, 29, 59]
    print("=== Piece 2: Median Gradient Norm Through Time (Markdown Table for README) ===")
    print("| Configuration | 0 steps back ($t=59$) | 9 steps back ($t=50$) | 29 steps back ($t=30$) | 59 steps back ($t=0$) |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    curves = {}
    for label, kind, bias, color in configs:
        med_norms = measure_input_gradients(
            kind=kind, gate_bias=bias, seq_len=seq_len, batch_size=16,
            n_inputs=8, hidden=32, seeds=seeds
        )
        curves[label] = (med_norms, color)
        vals = [med_norms[k].item() for k in report_steps]
        print(
            f"| {label} | `{vals[0]:.3e}` | `{vals[1]:.3e}` | `{vals[2]:.3e}` | `{vals[3]:.3e}` |"
        )

    # Plot on log scale
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    steps_back = list(range(seq_len))
    for label, (med_norms, color) in curves.items():
        ax.plot(steps_back, med_norms.numpy(), lw=2.2, color=color, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Steps back from the loss (0 = last step t=59, 59 = first step t=0)")
    ax.set_ylabel("Median input gradient L2 norm (log scale)")
    ax.set_title("Gradient Norm Through Time at Initialization (Median over 5 Seeds)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("gradients.png", dpi=150)
    plt.close(fig)
    print("\nSaved log-scale plot to gradients.png")


if __name__ == "__main__":
    run_gradients()