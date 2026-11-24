"""PINN Starter: 입문자를 위한 PyTorch 기반 Physics-Informed Neural Network 패키지."""

from .autograd import derivative
from .model import MLP
from .problems import DampedHarmonicOscillator, HeatEquation1D

__all__ = [
    "MLP",
    "derivative",
    "DampedHarmonicOscillator",
    "HeatEquation1D",
]
