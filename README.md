# What Recurrence Can and Cannot Remember

Measuring gradients through time and comparing recurrent cells on a delayed-recall task.

## Prediction (written before running the clipping comparison)

**What I expected clipping to do for the plain RNN at length 30, and why:**
I don't think turning gradient clipping on or off will help the plain RNN solve the task at a length of 30. Either way, its accuracy will probably stay stuck at a low level. 

*Reasoning:* Gradient clipping caps the global norm of the gradients during backpropagation (`nn.utils.clip_grad_norm_`), which prevents exploding gradients from causing high weight updates. However, on a 30-step delayed-recall task, the signal from step 0 to 29 fails because of vanishing gradients.