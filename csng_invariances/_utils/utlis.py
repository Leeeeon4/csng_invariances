import torch


def print_cuda():
    """Prints the current cuda memory status."""
    res = torch.cuda.memory_reserved(0)
    allo = torch.cuda.memory_allocated(0)
    print(
        f"Reserved memory: {res:,}\n"
        f"Allocated memory: {allo:,}\n"
        f"Free memory: {(res-allo):,}"
    )
