# strategies module containing trading strategy implementations.

from .base import Strategy
from .moving_average import MovingAverageCrossover

__all__ = ['Strategy', 'MovingAverageCrossover']
