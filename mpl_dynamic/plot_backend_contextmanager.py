# https://github.com/matplotlib/matplotlib/issues/26362#issuecomment-1671370539
from contextlib import AbstractContextManager
import matplotlib.pyplot as plt
class plot_backend(AbstractContextManager):
    """Context manager for switching matplotlib backend."""

    backend: str
    current_backend: str

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.current_backend = plt.get_backend()
        # available_backends = plt.rcsetup.all_backends
        # assert backend in available_backends

    def __enter__(self):
        plt.switch_backend(self.backend)

    def __exit__(self, *args):
        plt.switch_backend(self.current_backend)