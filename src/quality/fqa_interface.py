#src/quality/fqa_interface.py

from abc import ABC, abstractmethod
import numpy as np


class QualityResult:
    def __init__(self, is_acceptable: bool, reason: str):
        self.is_acceptable = is_acceptable
        self.reason = reason


class FaceQualityAssessor(ABC):
    @abstractmethod
    def assess(self, face: np.ndarray) -> QualityResult:
        """
        Checks blur, lighting, contrast
        """
        pass
