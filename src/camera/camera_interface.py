#src/camera/camera_interface.py

from abc import ABC, abstractmethod
import numpy as np


class Camera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Returns:
            Raw BGR frame (H x W x 3)
        """
        pass
