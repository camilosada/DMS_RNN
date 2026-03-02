import torch
import torch.nn as nn
import math


class DMSRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=10, tau=10):
        super().__init__()
        self.alpha = dt / tau  # The Euler integration step size
        self.hidden_size = hidden_size

        # Define the network's connections
        self.in2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2out = nn.Linear(hidden_size, 1)  # Continuous evidence readout

        # Variance scaling initialization for stability
        std_r = 0.8 / math.sqrt(hidden_size)
        nn.init.normal_(self.h2h.weight, mean=0.0, std=std_r)
        std_o = 1.0 / math.sqrt(hidden_size)
        nn.init.normal_(self.h2out.weight, mean=0.0, std=std_o)
        std_in = 1.0 / math.sqrt(input_size)
        nn.init.normal_(self.in2h.weight, mean=0.0, std=std_in)

        # Freeze the output weights and biases
        self.h2out.weight.requires_grad = False
        self.h2out.bias.requires_grad = False

    def forward(self, x):
        # x is the input sequence with shape: (batch_size, time_steps, input_size)
        batch_size, time_steps, _ = x.shape

        # Start with a silent network (firing rates = 0)
        r = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Loop over time to integrate the biological dynamics
        output = []
        for t in range(time_steps):
            drive = self.h2h(r) + self.in2h(x[:, t, :])
            r = (1 - self.alpha) * r + self.alpha * torch.tanh(drive)
            output.append(r)

        # The decision is the linear readout of the final network state
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        decision_variable = self.h2out(output)
        return decision_variable, output
