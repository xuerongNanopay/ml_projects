import torch


def save_checkpoint(state, filename):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return 1

def print_example():
    pass
