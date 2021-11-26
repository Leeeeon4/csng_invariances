import torch
from datetime import datetime


def print_cuda():
    """Prints the current cuda memory status."""
    res = torch.cuda.memory_reserved(0)
    allo = torch.cuda.memory_allocated(0)
    print(
        f"Reserved memory: {res:,}\n"
        f"Allocated memory: {allo:,}\n"
        f"Free memory: {(res-allo):,}"
    )


def string_time():
    """Returns formated sting of current time.

    Returns:
        str: Current date and time in %Y-%m-%d_%H:%M:%S format.
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
