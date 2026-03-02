import torch


def generate_dms_batch(
    batch_size,
    sample_dur=450,
    test_dur=450,
    delay_min=200,
    delay_max=600,
    noise_std=0.1,
):
    sample_dur = 450
    test_dur = 450
    max_delay = delay_max  # allocate for the longest possible delay
    max_time_steps = sample_dur + max_delay + test_dur

    input_size = 2  # Our 2-theta encoded 1D channel

    # Initialize the input tensor with zeros
    x = torch.zeros(batch_size, max_time_steps, input_size)

    # Randomly choose orientations: +1 (horizontal) or -1 (vertical)
    samples = torch.randint(0, 2, (batch_size, 1)) * 2 - 1
    tests = torch.randint(0, 2, (batch_size, 1)) * 2 - 1

    # Sample a different delay length for each trial in the batch
    delays = torch.randint(delay_min, delay_max + 1, (batch_size,))

    # Broadcast the sample and test pulses into their respective time windows
    for i in range(batch_size):
        d = delays[i].item()
        test_start = sample_dur + d
        test_end = test_start + test_dur

        x[i, :sample_dur, 0] = samples[i]
        x[i, test_start:test_end, 1] = tests[i]

    # Calculate the target: Match = +1, Non-Match = -1
    y = samples * tests
    # Add noise to the input
    x = x + torch.randn_like(x) * noise_std
    return x, y.float(), delays


def make_loss_mask(
    batch_size, delays, sample_dur=450, test_dur=450, delay_max=600
):
    total_steps = sample_dur + delay_max + test_dur
    mask = torch.zeros(batch_size, total_steps)
    for i in range(batch_size):
        test_start = sample_dur + delays[i].item()
        test_end = test_start + test_dur
        mask[i, test_start:test_end] = 1.0
    return mask  # shape: (batch, time)
