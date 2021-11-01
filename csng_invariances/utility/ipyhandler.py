"""Module for checking if notebook, ipykernel or pythonterminal is used.

This module is based on the stackoverflow post: 
https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook#39662359
"""

# %%
def isnotebook():
    """Check if code is run as notebook.

    Returns:
        bool: True if used as ipynb or # %% cell.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
