# What Recurrence Can and Cannot Remember

Measuring gradients through time and comparing recurrent cells on a delayed-recall task.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python benchmark.py      # piece 1: 5 configs x lengths 10, 30, 60 (seeds 0, 1, 2)
python gradients.py      # piece 2: gradient per step (median over 5 seeds), saves gradients.png
python fixes.py          # piece 3: forget-bias sweep and RNN clipping comparison
python timing.py         # piece 4: LSTM forward+backward time against sequence length
```

## Prediction (written before running the clipping comparison)

**What I expected clipping to do for the plain RNN at length 30, and why:**
I don't think turning gradient clipping on or off will help the plain RNN solve the task at a length of 30. Either way, its accuracy will probably stay stuck at a low level. 

*Reasoning:* Gradient clipping caps the global norm of the gradients during backpropagation (`nn.utils.clip_grad_norm_`), which prevents exploding gradients from causing high weight updates. However, on a 30-step delayed-recall task, the signal from step 0 to 29 fails because of vanishing gradients.

## Results

### Piece 1: Benchmark Table (Validation Accuracy; Chance = 0.250)

All runs use identical hyperparameters (`hidden=32`, `steps=600`, `batch_size=64`, `lr=3e-3`, `clip=1.0`, validation seed `999`).

| Configuration | Length | Seed 0 | Seed 1 | Seed 2 | Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Plain RNN | 10 | 0.755 | 1.000 | 0.495 | 0.750 |
| Plain RNN | 30 | 0.248 | 0.251 | 0.264 | 0.254 |
| Plain RNN | 60 | 0.250 | 0.240 | 0.233 | 0.241 |
| LSTM (default) | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM (default) | 30 | 0.248 | 0.251 | 0.255 | 0.251 |
| LSTM (default) | 60 | 0.249 | 0.251 | 0.254 | 0.251 |
| LSTM (forget bias 3) | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM (forget bias 3) | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM (forget bias 3) | 60 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU (default) | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU (default) | 30 | 0.241 | 0.257 | 0.254 | 0.251 |
| GRU (default) | 60 | 0.248 | 0.249 | 0.240 | 0.246 |
| GRU (update bias 3) | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU (update bias 3) | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU (update bias 3) | 60 | 1.000 | 1.000 | 1.000 | 1.000 |

### Piece 2: Gradient-Through-Time Measurement

Median L2 gradient norm $\left\|\frac{\partial \mathcal{L}}{\partial x_t}\right\|_2$ reaching each input step across 5 seeds (`0, 1, 2, 3, 4`) for a 60-step sequence (`hidden=32`, `batch=16`, `n_inputs=8`) when the cross-entropy loss reads only the final step ($t = 59$):

| Configuration | 0 steps back ($t=59$) | 9 steps back ($t=50$) | 29 steps back ($t=30$) | 59 steps back ($t=0$) |
| :--- | :---: | :---: | :---: | :---: |
| Plain RNN | `3.144e-02` | `8.030e-05` | `3.101e-10` | `2.000e-18` |
| LSTM (default, bias 0) | `9.021e-03` | `6.370e-05` | `5.646e-09` | `5.448e-15` |
| LSTM (forget bias 3) | `7.627e-03` | `3.223e-03` | `2.434e-03` | `5.871e-03` |

![Gradient Norm Through Time](gradients.png)

### Piece 3: Two Controlled Fixes (Sequence Length 30, Seeds 0, 1, 2)

#### 3A. LSTM Forget-Gate Bias Sweep ($T = 30$)

| Forget Gate Bias | Seed 0 | Seed 1 | Seed 2 | Mean |
| :--- | :---: | :---: | :---: | :---: |
| `bias = 0.0` (default) | 0.247 | 0.251 | 0.258 | 0.252 |
| `bias = 1.0` | 0.495 | 0.496 | 1.000 | 0.664 |
| `bias = 2.0` | 1.000 | 1.000 | 1.000 | 1.000 |
| `bias = 3.0` | 1.000 | 1.000 | 1.000 | 1.000 |

#### 3B. Plain RNN Gradient Clipping Comparison ($T = 30$)

| Clipping Setting | Seed 0 | Seed 1 | Seed 2 | Mean |
| :--- | :---: | :---: | :---: | :---: |
| Clipping ON (`clip=1.0`) | 0.248 | 0.251 | 0.264 | 0.254 |
| Clipping OFF (`clip=None`) | 0.248 | 0.251 | 0.264 | 0.254 |

**Explanation of Piece 3 Results:**
The clipping comparison confirms our pre-run prediction: the Plain RNN remains at chance level whether gradient clipping is enabled or disabled. Clipping scales down large gradients to prevent explosion, but it cannot recover a vanished gradient signal from the start of the sequence. Conversely, in the LSTM forget-bias sweep, default initialization fails on certain seeds, whereas opening the forget gate stabilizes gradient flow from the first update and achieves perfect accuracy across all seeds.

### Piece 4: Limits of Recurrence (Timing & Distance)

#### 4A. LSTM Forward + Backward Pass Timing (Best of 3 Repeats, Batch = 64, Hidden = 32)

| Sequence Length ($T$) | Best Time (ms) | Ratio vs. $T = 50$ |
| :---: | :---: | :---: |
| 50 | 4.63 | 1.00x |
| 100 | 8.79 | 1.90x |
| 200 | 17.34 | 3.74x |
| 400 | 35.60 | 7.68x |

#### 4B. Distance Limit Check (Open-Gate LSTM `bias = 3.0`, 600 Updates, Seed 0)

| Sequence Length ($T$) | Validation Accuracy (Seed 0) | Outcome |
| :---: | :---: | :--- |
| 60 | 1.000 | Solved |
| 100 | 0.252 | Chance (0.250) |
| 200 | 0.251 | Chance (0.250) |

## Conclusion

1. **What gates changed:** A plain RNN overwrites its hidden state at every step, causing gradients to vanish geometrically with distance. Gated architectures like LSTM and GRU introduce an additive state update path, creating a potential highway capable of carrying information and gradients across many steps.
2. **What initialization changed:** Having a gated path in the equations is not enough if the gate starts half-closed. At default initialization, the cell state is scaled down at every step, causing the gradient to vanish rapidly. Initializing the forget-gate bias to a higher value keeps the gate open from the first update, allowing the LSTM to achieve perfect accuracy across all lengths.
3. **What clipping changed:** Gradient clipping did not improve the plain RNN. Clipping bounds exploding gradients so updates do not diverge, but it cannot fix vanishing gradients.
4. **The two limits of recurrence and how direct connections (attention) address them:**
   - **Sequential bottleneck ($O(T)$ serial steps):** Because each step requires the hidden state of the previous step, computation cannot be parallelized across the sequence dimension, meaning doubling the length roughly doubles the processing time. Direct connections like self-attention remove recurrence so all positions are computed simultaneously in parallel.
   - **Distance limit:** Even with an open gate, the initial label must survive many intermediate state updates inside a fixed-width vector, causing performance to drop back to chance at very long sequence lengths. Direct connections link every position directly in a single hop so distance stops mattering.

## Known Limitations

- All training comparisons use a fixed budget of `600` updates, a single learning rate (`3e-3`), and hidden size `32`; an open-gate LSTM could potentially learn beyond $T = 60$ given substantially more updates or hyperparameter tuning.
- In a GRU, a large positive update-gate bias couples memory retention with input suppression, which can slightly dampen writing the initial input compared to an LSTM where gates are parameterized independently.