"""Core mathematical primitives for tropical SDR framework."""

from .tropical_ops import (
    tropical_add,
    tropical_mul,
    tropical_dot,
    tropical_distance,
    tropical_polynomial
)
from .sdr import SDR, SDRConfig
from .polytope import TropicalPolytope, Amoeba
from .padic import PadicTimer, PadicPhase

__all__ = [
    'tropical_add',
    'tropical_mul', 
    'tropical_dot',
    'tropical_distance',
    'tropical_polynomial',
    'SDR',
    'SDRConfig',
    'TropicalPolytope',
    'Amoeba',
    'PadicTimer',
    'PadicPhase'
]